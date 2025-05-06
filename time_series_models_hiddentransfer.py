from typing import Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import kgml_lib
# from sequence_dataset import SequenceDataset, train_test_split

import matplotlib.pyplot as plt

class SequenceDataset(Dataset):
    def __init__(self, inputs, outputs, sequence_length=365):
        """
        Args:
            inputs (np.ndarray): Array of shape (time_steps, num_sites, num_input_features)
            outputs (np.ndarray): Array of shape (time_steps, num_sites, num_output_features)
            sequence_length (int): Number of consecutive days for each sample.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.sequence_length = sequence_length
        
        self.samples = []
        num_sites = inputs.shape[0]
        num_days = inputs.shape[1]
        
        # For each site, create samples by sliding a window over time.
        # Each sample uses a window of length `sequence_length` as input
        # and the target is the corresponding window of outputs.
        for site in range(num_sites):
            site_input = inputs[site] # Shape [total_days, input_features]
            # site_output = outputs[site] # Shape [total_days, input_features]
            
            for start in range(0,len(site_input), sequence_length):
                end = start + sequence_length
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


class SequenceDataset_multiYears(Dataset):
    def __init__(self, inputs, outputs, sequence_length=365):
        """
        Args:
            inputs (np.ndarray): Array of shape (time_steps, num_sites, num_input_features)
            outputs (np.ndarray): Array of shape (time_steps, num_sites, num_output_features)
            sequence_length (int): Number of consecutive days for each sample.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.sequence_length = sequence_length
        
        self.samples = []
        self.num_sites = inputs.shape[0]
        self.num_days = inputs.shape[1]
        
    def __len__(self):
        return self.num_sites
    
    def __getitem__(self, index):
        # Get input sequence and corresponding output sequence.
        x_seq = self.inputs[index, :, :]  # (sequence_length, num_input_features)
        y_seq = self.outputs[index, :, :]   # (sequence_length, num_output_features)
        # Convert to torch tensors.
        if not isinstance(x_seq, torch.Tensor):
            x_seq = torch.tensor(x_seq, dtype=torch.float32)
            y_seq = torch.tensor(y_seq, dtype=torch.float32)
        return x_seq, y_seq
    
# A commom class for Time series Models
class TimeSeriesModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, sequence_length=365):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if output_dim > 1:
            self.compute_r2 = kgml_lib.R2Loss_mul()
        else:
            self.compute_r2 = kgml_lib.R2Loss()


        # self.model: nn.Module = ts_model
        # self.optimizer: optim.Optimizer = None
        
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epochs_no_improve = 0


    def train_test_split(self, X, Y, num_sites, num_years, train_batch_size, split_method = 'temporal', year_splitting=False):
        self.total_sites = num_sites
        self.total_years = num_years
        days_per_year = 365
        # total_days = total_years * days_per_year
        # num_sites = X.shape[0] #100

        # num_input_features = 19
        # num_output_features = 3


        # Define the training and test split:
        if split_method == 'temporal': # Choose last two yeas for test
            if num_years > 4:
                test_years = 2
            else:
                test_years = 1
            train_years = num_years - test_years
            train_days = train_years * days_per_year  # first 18 years for training
            self.train_years = train_years
            self.test_years = test_years
            # Split the data along the time dimension.
            X_train = X[:, :train_days, :]
            X_test = X[:, train_days:, :]

            Y_train = Y[:, :train_days, :]
            Y_test = Y[:, train_days:, :]

            # Y2_train = Y2[:,:train_years, :]
            # Y2_test = Y2[:, train_years:, :]
        elif split_method == 'spatial':
            train_sites = int(num_sites * 0.8)

            # Split the data along the time dimension.
            shuffled_ix = torch.randperm(self.X.size()[0])
            X = X[shuffled_ix,:,:]
            Y = Y[shuffled_ix,:,:]

            X_train = X[:train_sites, :, :]
            X_test = X[train_sites:, :, :]

            Y_train = Y[:train_sites, :, :]
            Y_test = Y[train_sites:, :, :]

            self.train_years = num_years
            self.test_years = num_years
        else:
            pass # Add code later

        
        # Create Dataset objects for training and testing.
        sequence_length = self.sequence_length  # Must Use 365 whole year's days as a sample
        if year_splitting:
            self.train_years = 1
            self.test_years = 1
            train_dataset = SequenceDataset(X_train, Y_train, sequence_length)
            test_dataset = SequenceDataset(X_test, Y_test, sequence_length)
        else:
            train_dataset = SequenceDataset_multiYears(X_train, Y_train, sequence_length)
            test_dataset = SequenceDataset_multiYears(X_test, Y_test, sequence_length)

        # Create DataLoaders.
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # return train_loader, test_loader


    def load_pretrained(self, model, pretrained_model_path=None):

        #output 4 in first module and 1 in second module
        checkpoint = torch.load(pretrained_model_path, weights_only=False, map_location=torch.device('mps'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(self.model)
        params = list(self.model.parameters())
        print(len(params))
        print(params[5].size())  # conv1's .weight
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def myloss_mul_sum(output, target, loss_weights):
        if not loss_weights:
            loss_weights = [1.0] * output
        mse_loss = nn.MSELoss(reduction='mean')  # Mean squared error
        total_loss = 0.0
        losses = []

        nout = output.size(2)  # Number of output features
        for i in range(nout):
            individual_loss = mse_loss(output[:, :, i], target[:, :, i])  # Compute MSE for each output
            weighted_loss = loss_weights[i] * individual_loss  # Apply weight
            total_loss += weighted_loss  # Sum up weighted losses
            losses.append(weighted_loss.item())  # Store individual loss values

        return total_loss, losses

    def train_model(self, LR=0.001,step_size=20, gamma=0.8, maxepoch=80):

        # model = self.model
        self.to(self.device)
        self.criterion = nn.MSELoss() # nn.L1Loss() # For regression
        # optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer = optim.Adam(self.parameters(), lr=LR)
        compute_r2 = self.compute_r2
        num_epochs = maxepoch  # Adjust as needed
        # step_size = 20
        # gamma = 0.8

        # For early stop of training
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 40
        checkpoint_path='best_GRU_model.pth'

        # StepLR scheduler 
        scheduler = StepLR(optimizer, step_size= step_size, gamma= gamma)

        for epoch in range(num_epochs):
            # model.train()
            self.train()
            train_losses = []
            train_predictions = []
            train_targets = []

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                if self.train_years > 1:  # (batch_size, num_years, 365, features)
                    total_loss = 0
                    hidden_state = None  # Initialize hidden state
                    outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                    for year in range(self.train_years):  # Loop over years
                        X_batch_year = batch_x[:, year*self.sequence_length:(year+1)*self.sequence_length, :]
                        Y_batch_year = batch_y[:, year*self.sequence_length:(year+1)*self.sequence_length, :]
                        if hidden_state is None:
                            outputs, hidden_state = self(X_batch_year)
                        else:
                            outputs, hidden_state = self(X_batch_year, hidden_state)

                        total_loss += self.criterion(outputs, Y_batch_year)
                        outputs_pred[:, year*self.sequence_length:(year+1)*self.sequence_length, :] = outputs[:,:,:]

                    hidden_state = tuple(h.detach() for h in hidden_state)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(total_loss.item() / self.train_years)  # Normalize loss per year
                else:  # Single-year data (batch_size, 365, features)
                    outputs_pred = self(batch_x)
                    loss = self.criterion(outputs_pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())
                train_predictions.append(outputs_pred.cpu())
                train_targets.append(batch_y.cpu())
            
            train_predictions = torch.cat(train_predictions, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_R2 = compute_r2(train_predictions, train_targets)
            avg_train_loss = np.mean(train_losses)

            # Evaluate on the test set.
            # model.eval()
            self.eval()
            test_losses = []
            test_predictions = []
            test_targets = []
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if self.test_years > 1:  # Multi-year input
                        total_loss = 0
                        hidden_state = None
                        outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                        for year in range(self.test_years):  
                            X_batch_year = batch_x[:, year*self.sequence_length:(year+1)*self.sequence_length, :]
                            Y_batch_year = batch_y[:, year*self.sequence_length:(year+1)*self.sequence_length, :]

                            if hidden_state is None:
                                outputs, hidden_state = self(X_batch_year)
                            else:
                                outputs, hidden_state = self(X_batch_year, hidden_state)

                            total_loss += self.criterion(outputs, Y_batch_year)

                        test_losses.append(total_loss.item() / self.test_years)
                        outputs_pred[:, year*self.sequence_length:(year+1)*self.sequence_length, :] = outputs[:,:,:]
                    else:  # Single-year input
                        outputs_pred = self(batch_x)
                        loss = self.criterion(outputs_pred, batch_y)
                        test_losses.append(loss.item())
                    test_predictions.append(outputs_pred.cpu())
                    test_targets.append(batch_y.cpu())
            
            test_predictions = torch.cat(test_predictions, dim=0)
            test_targets = torch.cat(test_targets, dim=0)
            test_R2 = compute_r2(test_predictions, test_targets)
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
            print(f'Train R2:  {train_R2:.2f}', f'Test R2:  {test_R2:.2f}')

    def test(self):
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
                outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                # Get model predictions
                # outputs_pred = model(batch_x)
                if self.test_years > 1:  # Multi-year input
                    total_loss = 0
                    hidden_state = None

                    for year in range(self.test_years):  
                        X_batch_year = batch_x[:, year*self.sequence_length:(year+1)*self.sequence_length, :]
                        Y_batch_year = batch_y[:, year*self.sequence_length:(year+1)*self.sequence_length, :]

                        if hidden_state is None:
                            outputs, hidden_state = self(X_batch_year)
                        else:
                            outputs, hidden_state = self(X_batch_year, hidden_state)

                        total_loss += self.criterion(outputs, Y_batch_year)

                    test_losses.append(total_loss.item() / self.test_years)
                    outputs_pred[:, year*self.sequence_length:(year+1)*self.sequence_length, :] = outputs[:,:,:]
                else:  # Single-year input
                    outputs_pred = self(batch_x)
                    loss = self.criterion(outputs_pred, batch_y)
                    test_losses.append(loss.item())

                # Compute loss
                loss = self.criterion(outputs_pred, batch_y)
                test_losses.append(loss.item())
                
                # (Optional) Save predictions and targets for further analysis
                all_predictions.append(outputs_pred.cpu())
                all_targets.append(batch_y.cpu())
        
        # Calculate the average test loss
        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # (Optional) If you want to concatenate all predictions and targets:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        compute_r2 = self.compute_r2
        test_R2 = compute_r2(all_predictions, all_targets)
        print(f'Test R2:  {test_R2:.2f}')

    def check_results(self, device, check_xset, check_yset, num_years):
        """
        check and record model output for later visualization (1 year only)
        check_x_set: the input that needs to check
        check_yset: the ground truth
        num_year: the number of year that need to record (from 0 to 17)
        """
        model_trained = self
        model_trained.to(device)
        model_trained.eval()

        check_pred = torch.zeros(check_yset.size(), device = device)
        with torch.no_grad():
            for year in range(num_years):  
                X_batch_year = check_xset[:,year*self.sequence_length:(year+1)*self.sequence_length,:].to(device)  # shape: (batch, days, features)

                if year == 0:
                    outputs, hidden_state = model_trained(X_batch_year)  
                else:
                    outputs, hidden_state = model_trained(X_batch_year, hidden_state) 
                check_pred[:,year*self.sequence_length:(year+1)*self.sequence_length,:] = outputs[:,:,:]

        return check_pred.cpu()


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


    def vis_scatter(self, Y_scaler=None):
        ########################################################################################
        ###visualize scatter plots of training and testing
        ########################################################################################

        # if data_path:
        #     data = torch.load(data_path,map_location=torch.device('cpu'),weights_only=False)
        # else:
        #     data = self.dataset

        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        Y_test = self.Y_test
        Y_train_pred = self.check_results(self.device, X_train, Y_train, self.train_years)
        Y_test_pred = self.check_results(self.device, X_test, Y_test, self.test_years)

        compute_r2 = self.compute_r2
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['xtick.labelsize']=13
        plt.rcParams['ytick.labelsize']=13
        ncols = 2
        nrows = Y_train.size(2)
        
        if Y_scaler is None:
            Y_scaler = np.ones((nrows,2))

        fig, axes = plt.subplots(nrows,ncols,figsize=(6*ncols, 5*nrows))   
        test_names = ['Train','Test']

        for i in range(nrows):
            for col in range(2):
    
                if col == 0:
                    Ysim = kgml_lib.Z_norm_reverse(Y_train_pred[:,:,i],Y_scaler[i,:]).to("cpu")
                    Yobs = kgml_lib.Z_norm_reverse(Y_train[:,:,i],Y_scaler[i,:]).to("cpu")
                    # Y_maskb = Y2_maskb_val[:,:,i].to("cpu")
                else:
                    Ysim = kgml_lib.Z_norm_reverse(Y_test_pred[:,:,i],Y_scaler[i,:]).to("cpu")
                    Yobs = kgml_lib.Z_norm_reverse(Y_test[:,:,i],Y_scaler[i,:]).to("cpu")
                    # Y_maskb = Y2_maskb_test[:,:,i].to("cpu")
                if i < 2:
                    Ysim = - Ysim
                    Yobs = - Yobs
                
                if nrows > 1:
                    ax = axes[i,col]
                else:
                    ax = axes[col]
                ax.scatter( Yobs.contiguous().view(-1).numpy(),Ysim.contiguous().view(-1).numpy(), s=10,color='black',alpha=0.5)
                R2 = compute_r2(Ysim, Yobs).numpy()
                RMSE =  np.sqrt(kgml_lib.my_loss(Ysim, Yobs).numpy())
                Bias = torch.mean(Ysim-Yobs).numpy()
                m, b, r_value, p_value, std_err = stats.linregress(Ysim.contiguous().view(-1).numpy(), Yobs.contiguous().view(-1).numpy()) #r,p,std
                # lim_min = ylims[i][0]
                # lim_max = ylims[i][1]
                lim_min = min(np.min(Ysim.contiguous().view(-1).numpy()), np.min(Yobs.contiguous().view(-1).numpy()))
                lim_max = max(np.max(Ysim.contiguous().view(-1).numpy()), np.max(Yobs.contiguous().view(-1).numpy()))
                ax.set_xlim([lim_min, lim_max])
                ax.set_ylim([lim_min, lim_max])
                ax.text(0.1,0.75,'R$^2$=%0.3f\nRMSE=%0.3f\nbias=%0.3f' 
                    % (R2,RMSE,Bias), transform=ax.transAxes, fontsize = 12)
                if i == nrows-1:
                    ax.set_xlabel('Synthetic',fontsize = 15,weight='bold')
                if col == 0:
                    ax.set_ylabel('Predicted',fontsize = 15,weight='bold')
                ax.set_title(test_names[col],fontsize = 15,weight='bold')
                ax.plot(Yobs, m*Yobs + b,color='steelblue',lw=1.0)
                ax.plot([lim_min,lim_max], [lim_min,lim_max],color='red',linestyle='--')
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
    
    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        gru_out, hidden = self.gru(x, hidden)
        out = self.fc(gru_out)
        return out, hidden
    
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
    
    
class N2OGRU_multitask(TimeSeriesModel):
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


# use Reco model v1--GRU model
class RecoGRU_multitask(TimeSeriesModel):
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
    
