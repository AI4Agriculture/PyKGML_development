from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
# from sequence_dataset import SequenceDataset, train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

def Z_norm_reverse(X,Xscaler,units_convert=1.0):
    return (X*Xscaler[1]+Xscaler[0])*units_convert

def Z_norm_with_scaler(X,Xscaler):
    return (X-Xscaler[0])/Xscaler[1]

class SequenceDataset(Dataset):
    def __init__(self, inputs, outputs, days_per_year=365):
        """
        Args:
            inputs (np.ndarray): Array of shape (num_sites, time_steps, num_input_features)
            outputs (np.ndarray): Array of shape (num_sites, time_steps, num_output_features)
            days_per_year (int): Number of consecutive days for each year.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.sequence_length = days_per_year
        
        self.samples = []
        num_sites = inputs.shape[0]
        num_days = inputs.shape[1]
        
        # For each site, create samples by sliding a window over time.
        # Each sample uses a window of length `sequence_length` as input
        # and the target is the corresponding window of outputs.
        for site in range(num_sites):
            site_input = inputs[site] # Shape [total_days, input_features]
            # site_output = outputs[site] # Shape [total_days, input_features]
            
            for start in range(0,len(site_input), self.sequence_length):
                end = start + self.sequence_length
                if end > len(site_input):
                    break  # Discard incomplete sequences
                else:
                    self.samples.append((site, start, end))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        site, start, end = self.samples[index]
        # Get input sequence and corresponding output sequence.
        x_seq = self.inputs[site, start:end, :]  # (sequence_length, num_input_features)
        y_seq = self.outputs[site, start:end, :]   # (sequence_length, num_output_features)
        # Convert to torch tensors.
        if not isinstance(x_seq, torch.Tensor):
            x_seq = torch.tensor(x_seq, dtype=torch.float32)
            y_seq = torch.tensor(y_seq, dtype=torch.float32)
        return x_seq, y_seq

# A commom class for Time series Models
class TimeSeriesModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # self.model: nn.Module = ts_model
        # self.optimizer: optim.Optimizer = None
        
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epochs_no_improve = 0


    def train_test_split(self,X, Y, total_years,train_batch_size, days_per_year:int = 365, split_method:int = 0):
        '''
        split_method: int
            0: choose the last two years  
            1: randomly pick two sites

        '''
        # total_years = 18
        
        # total_days = total_years * days_per_year
        # num_sites = X.shape[0] #100

        # num_input_features = 19
        # num_output_features = 3


        # Define the training and test split:
        if split_method == 0: # Choose last two yeas for test
            train_years = total_years - 2
            train_days = train_years * days_per_year  # first 18 years for training

            # Split the data along the time dimension.
            X_train = X[:, :train_days, :]
            X_test = X[:, train_days:, :]

            Y_train = Y[:, :train_days, :]
            Y_test = Y[:, train_days:, :]

            # Y2_train = Y2[:,:train_years, :]
            # Y2_test = Y2[:, train_years:, :]

        else: # 1 Random pick two sites
            site_num = X.shape[0]
            site_random_list = np.random.choice(site_num, size=site_num, replace=False)

            test_site_num = max(int(site_num * 0.2), 1)

            train_site = site_random_list[:test_site_num]
            test_site = site_random_list[test_site_num:]

            X_train = X[train_site, :, :]
            X_test = X[test_site, :, :]

            Y_train = Y[train_site, :, :]
            Y_test = Y[test_site, :, :]

        
        # Create Dataset objects for training and testing.
        sequence_length = days_per_year  # Must Use whole year's days as a sequence

        train_dataset = SequenceDataset(X_train, Y_train, sequence_length)
        test_dataset = SequenceDataset(X_test, Y_test, sequence_length)

        # Create DataLoaders.
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False) # Note: Test batch size is 1

        # return train_loader, test_loader

    def train_model(self, loss_fun, LR=0.001,step_size=20, gamma=0.8, maxepoch=80, use_y_mask = False):

        # Initial parameters
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0

        self.to(self.device)
        self.criterion = loss_fun # nn.L1Loss() # nn.MSELoss() # For regression
        # optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer = optim.Adam(self.parameters(), lr=LR)

        num_epochs = maxepoch  # Adjust as needed
        # step_size = 20
        # gamma = 0.8

        # For early stop of training
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 20
        checkpoint_path='best_GRU_model.pth'

        # StepLR scheduler 
        scheduler = StepLR(optimizer, step_size= step_size, gamma= gamma)

        for epoch in range(num_epochs):
            # model.train()
            self.train()
            train_losses = []
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                # outputs_pred = model(batch_x)  # shape: (batch_size, sequence_length, output_dim)
                outputs_pred = self(batch_x)  # shape: (batch_size, sequence_length, output_dim)
                if use_y_mask:
                    y_mask = batch_y[..., 5:]
                    y_true = batch_y[..., :5] * y_mask
                    y_pred = outputs_pred * y_mask
                    loss = self.criterion(y_pred, y_true)
                else:
                    loss = self.criterion(outputs_pred, batch_y)

                loss.backward()
                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)

            # Evaluate on the test set.
            # model.eval()
            self.eval()
            test_losses = []
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs_pred = self(batch_x)

                    if use_y_mask:
                        y_mask = batch_y[..., 5:]
                        y_true = batch_y[..., :5] * y_mask
                        y_pred = outputs_pred * y_mask
                        loss = self.criterion(y_pred, y_true)
                    else:
                        loss = self.criterion(outputs_pred, batch_y)
                    
                    test_losses.append(loss.item())
            
            avg_test_loss = np.mean(test_losses)

            # Step the scheduler after each epoch
            scheduler.step()
            
            # Early stopping check
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                # torch.save(model.state_dict(), checkpoint_path)
                torch.save(self.state_dict(), checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= 40:
                    print(f'\n#***# Early stopping triggered after {epoch+1} epochs!')
                    # model.load_state_dict(torch.load(checkpoint_path))
                    self.load_state_dict(torch.load(checkpoint_path))
                    break
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_test_loss)
            self.epochs = epoch+1
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    def test(self, use_y_mask=False):
        # model = self.model
        # model.eval()
        self.eval()

        test_losses = []
        all_predictions = []  # Optional: to store predictions for further analysis
        all_targets = []      # Optional: to store true targets
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                #print("Count = ", count)
                # Move the batch data to the proper device (GPU or CPU)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get model predictions
                # outputs_pred = model(batch_x)
                outputs_pred = self(batch_x)

                # Compute loss
                if use_y_mask:
                    y_mask = batch_y[..., 5:]
                    y_true = batch_y[..., :5] * y_mask
                    y_pred = outputs_pred * y_mask
                else:
                    y_true = outputs_pred
                    y_pred = batch_y
                
                loss = self.criterion(y_pred, y_true)
                test_losses.append(loss.item())
                
                # Save predictions and targets for further analysis
                all_predictions.append(y_pred.cpu())
                all_targets.append(y_true.cpu())

        # Calculate the average test loss
        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Save true value and prediction for next step calculation :
        self.all_predictions = torch.cat(all_predictions, dim=0)
        self.all_targets = torch.cat(all_targets, dim=0)

    def get_R2_score(self, y_scaler:list, output_feature_name:list):
        y_param_num = self.all_targets.shape[-1]
        all_predictions_flat = self.all_predictions.reshape(-1,y_param_num)
        all_targets_flat = self.all_targets.reshape(-1,y_param_num)
        for i in range(y_param_num):
            _r2 = r2_score(Z_norm_reverse(all_targets_flat[:,i], y_scaler[i]), Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i]))
            print(f"Feature {output_feature_name[i]} R2 Score is: {_r2}")

    def plot_training_curves(self):
        epoch_list = np.arange(1, self.epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, self.train_losses, label='Training Loss')
        plt.plot(epoch_list, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def Vis_prediction_result(self, y_scaler:list, features:list, site:int,year:int):
        y_param_num = self.all_targets.shape[-1]
        # self.all_predictions shape is [200, 365, features]
        _idx = site*2 + year
        all_predictions_flat = self.all_predictions[_idx, :, :].reshape(-1,y_param_num)
        all_targets_flat     = self.all_targets[_idx,:,:].reshape(-1,y_param_num)
        
        N, F = all_targets_flat.shape # N: 365, F: features number

        fig, axes = plt.subplots(F, 2, figsize=(12, 4*F))

        for i, name in enumerate(features):
            ax_line, ax_scatter = axes[i]

            y_true = Z_norm_reverse(all_targets_flat[:,i], y_scaler[i]).numpy()
            y_pred = Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i]).numpy()
            
            # 1) LINE PLOT: Days index vs. values
            ax_line.plot(np.arange(N), y_true, label='True', lw=1)
            ax_line.plot(np.arange(N), y_pred, label='Pred', lw=1, alpha=0.7)
            ax_line.set_title(f"{name} over Days")
            ax_line.set_xlabel("Days Index")
            ax_line.set_ylabel(name)
            ax_line.legend(fontsize=8)

            # 2) SCATTER PLOT: true vs. pred
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())

            m, b, r_value, p_value, std_err = stats.linregress(y_true, y_pred) #r,p,std
            ax_scatter.plot(y_pred, m*y_pred + b,color='red',lw=1.0)
            ax_scatter.plot([mn, mx], [mn, mx],color='steelblue',linestyle='--')

            # Compute metrics
            r2   = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            bias = np.mean(y_pred - y_true)

            # Format text
            stats_text = (
                f"$R^2$ = {r2:.3f}\n"
                f"RMSE = {rmse:.3f}\n"
                f"Bias = {bias:.3f}"
            )

            # Place text in the upper left in axes-fraction coordinates
            ax_scatter.text(
                0.05, 0.95, stats_text,
                transform= ax_scatter.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )

            ax_scatter.scatter(y_true, y_pred, alpha=0.4, s=10)
            # ax_scatter.plot([mn, mx], [mn, mx], 'k--', lw=1)
            ax_scatter.set_title(f"{name}: True vs. Predicted")
            ax_scatter.set_xlabel("True Value")
            ax_scatter.set_ylabel("Predicted Value")
            ax_scatter.set_xlim(mn, mx)
            ax_scatter.set_ylim(mn, mx)

        # Tighten up and show
        sub_title = f"Site {site} Year {year}"
        plt.suptitle(sub_title, fontsize=12)
        plt.tight_layout()
        plt.show()

# A GRU with Attension Regression Model
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, gru_out):
        # gru_out shape: (batch_size, seq_len, hidden_dim)
        Q = self.query(gru_out)  # Query projections
        K = self.key(gru_out)    # Key projections
        V = self.value(gru_out)  # Value projections
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector
        context = torch.bmm(attention_weights, V)
        
        # Combine with original GRU output
        combined = torch.cat([gru_out, context], dim=-1)
        return combined

class GRUSeq2SeqWithAttention(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, output_dim)  # Double input size due to concatenation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # gru_out shape: (batch_size, seq_len, hidden_dim)
        gru_out = self.dropout(gru_out)
        
        # Apply attention
        attended = self.attention(gru_out)  # (batch_size, seq_len, 2*hidden_dim)
        
        # Final projection
        out = self.fc(attended)  # (batch_size, seq_len, output_dim)
        return out
    
# A Simple GRU Regression model
class GRUSeq2Seq(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out)
        return out
    
# A Simple LSTM Regression model
class LSTMSeq2Seq(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            output_dim (int): Number of output features per time step.
        """
        super().__init__(input_dim, hidden_dim, num_layers, output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Apply a linear layer to every time step.
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        out = self.fc(lstm_out)     # out shape: (batch_size, sequence_length, output_dim)
        return out
    
