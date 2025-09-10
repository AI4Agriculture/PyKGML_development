from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
# from sequence_dataset import SequenceDataset, train_test_split

import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

import inspect

def Z_norm_reverse(X,Xscaler,units_convert=1.0):
    return (X*Xscaler[1]+Xscaler[0])*units_convert

def Z_norm_with_scaler(X,Xscaler):
    return (X-Xscaler[0])/Xscaler[1]

def plot_result(y_scaler:list, features:list, all_predictions_flat:list,all_targets_flat:list, site:int, year:int, sub_title:str=None):

    N, F = all_targets_flat.shape # N: 365, F: features number

    fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

    for i, name in enumerate(features):
        ax_line = axes[i]

        y_true = Z_norm_reverse(all_targets_flat[:,i], y_scaler[i])
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        y_pred = Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i])
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        
        # LINE PLOT: Days index vs. values
        ax_line.plot(np.arange(N), y_true, 'o', label='True', color='steelblue', lw=1)
        ax_line.plot(np.arange(N), y_pred, label='Pred', color='red',lw=1, alpha=0.7)
        ax_line.set_title(f"{name} over Days")
        ax_line.set_xlabel("Days Index")
        ax_line.set_ylabel(name)
        ax_line.legend(fontsize=8)

    # Tighten up and show
    if sub_title is None:
        full_title = f"Site {site} Year {year}"
    else:
        full_title = f"{sub_title} Site {site} Year {year}"
    fig.suptitle(full_title, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.show()

def scatter_result(y_scaler:list, features:list, all_predictions_flat, all_targets_flat,sub_title:str=None):

    N, F = all_targets_flat.shape # N: 365, F: features number

    fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

    for i, name in enumerate(features):
        ax_scatter = axes[i]

        y_true = Z_norm_reverse(all_targets_flat[:,i], y_scaler[i])
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        y_pred = Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i])
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()

        # SCATTER PLOT: true vs. pred
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
    if sub_title is None:
        full_title = "True vs Prediction Values"
    else:
        full_title = f"{sub_title} True vs Prediction Values"
    fig.suptitle(full_title, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.show()

''' Begin for Machine Learning Models. Those functions are for easy to train, test, and Visualiztion '''

def ML_train_and_test(regressors:dict, X_train, X_test, y_train, y_test, output_features:list):
    mean_scores = {}
    feature_scores = []
    prediction_results = dict()
    prediction_results['Ground Truth'] = y_test

    p_result = list()
    # Loop, fit, predict, score ---
    for name, reg in regressors.items():
        # if multi‐output, wrap in MultiOutputRegressor
        print(name)
        _p_result = dict()
        model = MultiOutputRegressor(reg)
        # train
        model.fit(X_train, y_train)
        # predict
        y_pred = model.predict(X_test)
        _p_result[name] = y_pred
        p_result.append(_p_result)
        # metrics
        r2   = r2_score(y_test, y_pred, multioutput="uniform_average")
        rmse = mean_squared_error(y_test, y_pred, multioutput="uniform_average")
        mae  = mean_absolute_error(y_test, y_pred, multioutput="uniform_average")
        mean_scores[name] = (r2, rmse, mae)

        for idx,feature_name in enumerate(output_features):
            scores = dict()
            r2 = r2_score(y_test[:,idx], y_pred[:,idx])
            rmse = mean_squared_error(y_test[:,idx], y_pred[:,idx])
            mae  = mean_absolute_error(y_test[:,idx], y_pred[:,idx])
            scores['Method'] = name
            scores['Feature'] = feature_name
            scores['R2'] = r2
            scores['RMSE'] = rmse
            scores['MAE'] = mae
            feature_scores.append(scores)

    prediction_results['Prediction'] = p_result

    return mean_scores, feature_scores, prediction_results

def ML_display_scores(mean_scores:dict, feature_scores:list):
    print("  Mean Scores   ")
    # Print a summary table ---
    print(f"{'Model':<20}   {'R2':>6}   {'RMSE':>8}   {'MAE':>8}")
    print("-"*48)
    for name,(r2,rmse,mae) in mean_scores.items():
        print(f"{name:<20}   {r2:6.3f}   {rmse:8.3f}   {mae:8.3f}")

    print("      ")
    print("  Each output feature's Scores   ")
    p_features_scores = pd.DataFrame(feature_scores)
    print(p_features_scores)

def ML_vis_prediction_results(prediction_results, features, y_scaler, sites:int, years:int, day_of_year:int, choiced_site:int, choiced_year:int):
    assert(choiced_site < sites)
    assert(choiced_year < years)
    
    y_test = prediction_results['Ground Truth']
    all_pred = prediction_results['Prediction']

    for item in all_pred:
        for method, y_pred in item.items():
            _y_pred = y_pred.reshape(sites, years,day_of_year,-1) # Reshape to [sites, years, day of year, output features]
            _y_test = y_test.reshape(sites, years,day_of_year,-1)
            
            site_idx = choiced_site
            year_idx = choiced_year
            all_predictions_flat = _y_pred[site_idx, year_idx,:,:]
            all_targets_flat     = _y_test[site_idx, year_idx,:,:]

            plot_result(y_scaler, features, all_predictions_flat,all_targets_flat, site=site_idx, year=year_idx, sub_title=method)
            scatter_result(y_scaler, features, y_pred, y_test, sub_title=method)

''' End of Machine Learning Models'''

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

    def load_pretrained(self, pretrained_model_path=None):
        checkpoint = torch.load(pretrained_model_path, weights_only=True)
        self.load_state_dict(checkpoint)
        
        print(self)
        params = list(self.parameters())
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())


    def train_model(self, loss_func, LR=0.001,step_size=20, gamma=0.8, maxepoch=80, use_y_mask = False, checkpoint_path='best_model.pth'):
        # Initial parameters
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0

        y_num = self.output_dim  # output features number

        self.to(self.device)
        self.criterion = loss_func # nn.L1Loss() # nn.MSELoss() # For regression
        # check if customized loss function
        sig = inspect.signature(loss_func.forward)
        num_params = len(sig.parameters)
        if num_params == 3: # customized loss function
            self.b_customized_loss = True
        elif num_params == 2:  # pytorch loss function, like nn.MSELoss
            self.b_customized_loss = False
        else:
            raise ValueError("The loss function error.")

        optimizer = optim.Adam(self.parameters(), lr=LR)

        num_epochs = maxepoch  # Adjust as needed
        # step_size = 20
        # gamma = 0.8

        # For early stop of training
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 20
        # checkpoint_path='best_model.pth'

        # StepLR scheduler 
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

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
                    y_mask = batch_y[..., y_num:]
                    y_true = batch_y[..., :y_num] * y_mask
                    y_pred = outputs_pred * y_mask
                else:
                    y_true = batch_y
                    y_pred = outputs_pred

                if self.b_customized_loss == True:
                    loss = self.criterion(y_pred, y_true, batch_x)
                else:
                    loss = self.criterion(y_pred, y_true)

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)

            # Evaluate on the test set.
            self.eval()
            test_losses = []
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs_pred = self(batch_x)

                    if use_y_mask:
                        y_mask = batch_y[..., y_num:]
                        y_true = batch_y[..., :y_num] * y_mask
                        y_pred = outputs_pred * y_mask
                    else:
                        y_true = batch_y
                        y_pred = outputs_pred
                    
                    if self.b_customized_loss == True:
                        loss = self.criterion(y_pred, y_true, batch_x)
                    else:
                        loss = self.criterion(y_pred, y_true)

                    test_losses.append(loss.item())
            
            avg_test_loss = np.mean(test_losses)

            # Step the scheduler after each epoch
            scheduler.step()
            
            # Early stopping check
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(self.state_dict(), checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= 40:
                    print(f'\n#***# Early stopping triggered after {epoch+1} epochs!')
                    self.load_state_dict(torch.load(checkpoint_path))
                    break
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_test_loss)
            self.epochs = epoch+1
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    def test(self, use_y_mask=False):
        self.eval()

        test_losses = []
        all_predictions = []  # Optional: to store predictions for further analysis
        all_targets = []      # Optional: to store true targets

        y_num= self.output_dim  # Output features number
        
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
                    y_mask = batch_y[..., y_num:]
                    y_true = batch_y[..., :y_num] * y_mask
                    y_pred = outputs_pred * y_mask
                else:
                    y_true = batch_y
                    y_pred = outputs_pred
                
                if self.b_customized_loss == True:
                    loss = self.criterion(y_pred, y_true, batch_x)
                else:
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

    # Select one site, one year in the test dataset
    # To plot curves for comparing the prediction and true values
    def Vis_plot_prediction_result_time_series(self, y_scaler:list, features:list, site:int,year:int):
        y_param_num = self.all_targets.shape[-1]
        # self.all_predictions shape is [200, 365, features]
        _idx = site*2 + year
        all_predictions_flat = self.all_predictions[_idx, :, :].reshape(-1,y_param_num)
        all_targets_flat     = self.all_targets[_idx,:,:].reshape(-1,y_param_num)
        plot_result(y_scaler, features, all_predictions_flat, all_targets_flat, site, year)
        

    def Vis_plot_prediction_result_time_series_masked(self, y_scaler:list, features:list, site:int,year:int, obs_mask:np.ndarray=None):
        y_param_num = self.all_targets.shape[-1]
        # self.all_predictions shape is [200, 365, features]
        _idx = site*2 + year
        predictions_flat = self.all_predictions[_idx, :, :].reshape(-1,y_param_num)
        targets_flat     = self.all_targets[_idx,:,:].reshape(-1,y_param_num)
        mask = obs_mask.reshape(self.all_targets.shape)[_idx,:,:].reshape(-1,y_param_num)
        N, F = targets_flat.shape # N: 365, F: features number

        fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

        for i, name in enumerate(features):
            ax_line = axes[i]

            y_true = Z_norm_reverse(targets_flat[:,i], y_scaler[i]).numpy()
            y_pred = Z_norm_reverse(predictions_flat[:,i], y_scaler[i]).numpy()
            if mask is not None or mask.size != 0:
                y_mask = mask.reshape(-1, mask.shape[-1])[:,i]
                y_true = np.where(y_mask == 0, np.nan, y_true)
            
            # LINE PLOT: Days index vs. values
            ax_line.plot(np.arange(N), y_true, 'o', label='True', color='steelblue', lw=1)
            ax_line.plot(np.arange(N), y_pred, label='Pred', color='red',lw=1, alpha=0.7)
            ax_line.set_title(f"{name} over Days")
            ax_line.set_xlabel("Days Index")
            ax_line.set_ylabel(name)
            ax_line.legend(fontsize=8)

        # Tighten up and show
        sub_title = f"Site {site} Year {year}"
        fig.suptitle(sub_title, fontsize=12)
        fig.subplots_adjust(top=0.9, hspace=0.4)
        plt.show()
            

    # Scatter the prediction value and true values base on test dataset
    def Vis_scatter_prediction_result(self, y_scaler:list, features:list):
        y_param_num = self.all_targets.shape[-1]
        # self.all_predictions shape is [200, 365, features]

        all_predictions_flat = self.all_predictions.reshape(-1,y_param_num)
        all_targets_flat     = self.all_targets.reshape(-1,y_param_num)
        
        scatter_result(y_scaler, features, all_predictions_flat, all_targets_flat)


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
    
### 1D CNN Regression models

# ==============================
# 1D CNN Model, Time series regression
# ==============================
class TemporalCNN(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        # Assume seq_len is 365
        self.cnn = nn.Sequential(
            # Block 1: Output is (batch_size, 32, seq_len)
            nn.Conv1d(input_dim, 32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Block 2: Output is (batch_size, 64, seq_len)
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Block 3: Output is (batch_size, 3, seq_len)
            nn.Conv1d(64, output_dim, kernel_size=1)
        )

    def forward(self, x):
        # CNN requires dimmesion as: (batch, channels, seq)
        x = x.permute(0, 2, 1)  # -> (batch, input_features, 365)
        out = self.cnn(x)          # (batch, 3, 365)
        return out.permute(0, 2, 1)  # change back to (batch, 365, 3)
    
class CNNLSTM(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        
        # 1D CNN Block: Extract local time series features
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM Block: Extract long time features
        self.lstm = nn.LSTM(
            input_size=128,   # input dim = CNN output channels
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True  # output is (batch, seq, features)
        )
        
        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),  # bi-direction LSTM *2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Origin x shape is: (batch_size, 365, input_features)
        # Change shape for Conv1d: (batch, channels, seq=365)
        x = x.permute(0, 2, 1)  # -> (batch, input_features, 365)
        
        # 1D CNN
        cnn_out = self.cnn(x)  # (batch, 128, 365)
        
        # change shape for LSTM: (batch, seq, features)
        lstm_input = cnn_out.permute(0, 2, 1)  # -> (batch, 365, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)  # (batch, 365, 512)  (bi-direction, hidden_size*2)
        
        # FC layer
        output = self.fc(lstm_out)  # (batch, 365, 3)
        return output
    
class CNN_LSTM_Attension(CNNLSTM):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2,attn_heads=8, attn_dropout=0.1):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        # Attension block after LSTM
        embed_size = self.lstm.hidden_size*2
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, 
                                               num_heads=attn_heads,
                                               dropout=attn_dropout,
                                                batch_first=True)

        # LayerNorm after the Attension
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Input x shape: (batch_size, 365, 16)
        # Change shape for Conv1d: (batch, channels, seq)
        x = x.permute(0, 2, 1)  # -> (batch, 16, 365)
        
        # 1D CNN
        cnn_out = self.cnn(x)  # (batch, 128, 365)
        
        # Change shape for LSTM: (batch, seq, features)
        lstm_input = cnn_out.permute(0, 2, 1)  # -> (batch, 365, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)  # (batch, 365, 512)  (bi-direction hidden_size*2)

        # Attension
        attn_out, _ = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            need_weights=False
        )

        attn_out = self.norm(attn_out + lstm_out)  # Residual connection

        # FC layer
        output = self.fc(attn_out)  # (batch, 365, 3)
        return output
    

        
# the 2-cell N2O KGML model from Licheng's 2022 paper
class N2OGRU_KGML(TimeSeriesModel):
    #input model variables are for each module
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim1, output_dim2, dropout):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim=output_dim1+output_dim2)
        if num_layers > 1:
            self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
            self.gru2 = nn.GRU(input_dim+output_dim1, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        else:
            self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.gru2 = nn.GRU(input_dim+output_dim1, hidden_dim, num_layers, batch_first=True)
        self.densor1 = nn.Linear(hidden_dim, output_dim1)
        self.densor2 = nn.Linear(hidden_dim, output_dim2)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 #may change to a small value
        self.densor1.bias.data.zero_()
        self.densor1.weight.data.uniform_(-initrange, initrange)
        self.densor2.bias.data.zero_()
        self.densor2.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(inputs.size(0))
    
        output1, hidden1 = self.gru1(inputs, hidden[0])
        output1 = self.densor1(self.drop(output1)) 
        inputs = torch.cat((inputs,output1),2)
        output2, hidden2 = self.gru2(inputs, hidden[1])
        output2 = self.densor2(self.drop(output2)) 
        #need to be careful what is the output orders!!!!!!!!!!!!!
        output=torch.cat((output2,output1),2)
        hidden=(hidden1,hidden2)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_dim),\
                weight.new_zeros(self.num_layers, bsz, self.hidden_dim))


# the Ra-Rh CO2 KGML model from Licheng's 2024 paper
class RecoGRU_KGML(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.GPP_scaler = GPP_scaler
        # self.Ra_scaler = Ra_scaler
        # self.Yield_scaler = Yield_scaler
        # self.Res_scaler = Res_scaler
        # sequence_length = self.sequence_length
        self.gru_basic = nn.GRU(input_dim,hidden_dim,2,dropout=dropout, batch_first=True)
        self.gru_Ra = nn.GRU(input_dim+hidden_dim, hidden_dim,1,batch_first=True)
        self.gru_Rh = nn.GRU(input_dim+hidden_dim+1, hidden_dim,2,dropout=dropout, batch_first=True)#+1 means res ini 
        self.gru_NEE = nn.GRU(input_dim+2, hidden_dim,1, batch_first=True)#+2 Ra and Rh
        self.drop=nn.Dropout(dropout)
        self.densor_Ra = nn.Linear(hidden_dim, 1)
        self.densor_Rh = nn.Linear(hidden_dim, 1)
        self.densor_NEE = nn.Linear(hidden_dim, 1)
        #attn for yield prediction
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yield = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 #may change to a small value
        self.densor_Ra.bias.data.zero_()
        self.densor_Ra.weight.data.uniform_(-initrange, initrange)
        self.densor_Rh.bias.data.zero_()
        self.densor_Rh.weight.data.uniform_(-initrange, initrange)
        self.densor_NEE.bias.data.zero_()
        self.densor_NEE.weight.data.uniform_(-initrange, initrange)
        for ii in range(4):
            self.attn[ii*2].bias.data.zero_()
            self.attn[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yield[ii*2].bias.data.zero_()
            self.densor_yield[ii*2].weight.data.uniform_(-initrange, initrange)
        

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(inputs.size(0))
    
        output, hidden1 = self.gru_basic(inputs, hidden[0])
        #predict yield
        inputs2 = self.drop(output)
        attn_weights = F.softmax(self.attn(inputs2), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs2 = torch.bmm(attn_weights,inputs2)
        output2 = self.densor_yield(inputs2)
        
        #predict flux
        #Ra
        output1 , hidden2 = self.gru_Ra(torch.cat((self.drop(output),inputs), 2), hidden[1])
        Ra = self.densor_Ra(self.drop(output1))
        
        #Rh
        #caculate annual Res
        #Annual GPP+ Annual Ra - Yield, GPP is no.8 inputs
        # annual_GPP = torch.sum(Z_norm_reverse(inputs[:,:,8],self.GPP_scaler),dim=1).view(-1,1,1)
        # annual_Ra = torch.sum(Z_norm_reverse(Ra[:,:,0],self.Ra_scaler),dim=1).view(-1,1,1)
        # annual_Yield = Z_norm_reverse(output2[:,0,0],self.Yield_scaler).view(-1,1,1)
        # #control 0< Res_ini < GPP
        # Res_ini = self.ReLU(annual_GPP+annual_Ra - annual_Yield)
        # #Res_ini[Res_ini > annual_GPP].data = annual_GPP[Res_ini > annual_GPP].data 
        # #scale Res_ini
        # Res_ini = Z_norm_with_scaler(Res_ini,self.Res_scaler)
        # ##calculate Rh now with current year res
        # Res = Res_ini.repeat(1, self.sequence_length, 1)
        # #left day 300
        # Res[:,0:298,:] = 0.0
        # Res[:,300:,:] = 0.0
        # output1, hidden3  = self.gru_Rh(torch.cat((Res,self.drop(output),inputs), 2), hidden[2])
        output1, hidden3  = self.gru_Rh(torch.cat((self.drop(output),inputs), 2), hidden[2])
        Rh = self.densor_Rh(self.drop(output1))
        
        #NEE
        output1, hidden4 = self.gru_NEE(torch.cat((Ra,Rh,inputs), 2), hidden[3])
        NEE = self.densor_NEE(self.drop(output1))
        output1 = torch.cat((Ra,Rh,NEE),2)
        
        return output1, output2, (hidden1,hidden2,hidden3,hidden4)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(2, bsz, self.hidden_dim),
                weight.new_zeros(1, bsz, self.hidden_dim),
                weight.new_zeros(2, bsz, self.hidden_dim),
                weight.new_zeros(1, bsz, self.hidden_dim))
    
# Transformer model
class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        pos_enc = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_enc  # (seq_len, d_model)

def generate_causal_mask(query_len, key_len, device):
    mask = torch.full((query_len, key_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle
    return mask

class TimeSeriesTransformer(TimeSeriesModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, d_model=128,  nhead=8, dropout=0.1):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim)
        self.input_proj = nn.Linear(input_dim, d_model)

        self.positional_encoding = RelPositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, time, features)
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)  # (batch, time, d_model)

        pos_enc = self.positional_encoding(seq_len).unsqueeze(0)  # (1, time, d_model)
        x = x + pos_enc  # broadcasting position encoding
        mask = generate_causal_mask(seq_len, seq_len, x.device) 

        # Transformer expects (batch, seq_len, d_model)
        out = self.transformer_encoder(x)


        out = self.output_proj(out)  # (batch, time, output_dim)
        return out