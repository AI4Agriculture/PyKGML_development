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
from time_series_models import TimeSeriesModel,SequenceDataset

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

def Z_norm_reverse(X,Xscaler,units_convert=1.0):
    return (X*Xscaler[1]+Xscaler[0])*units_convert

def Z_norm_with_scaler(X,Xscaler):
    return (X-Xscaler[0])/Xscaler[1]

# Get one site's all years data
class SequenceDataset_multiYears(Dataset):
    def __init__(self, inputs, outputs, sequence_length=365):
        """
        Args:
            inputs (np.ndarray): Array of shape (num_sites, years*days_of_year, num_input_features)
            outputs (np.ndarray): Array of shape (num_sites, years*days_of_year,num_output_features)
            sequence_length (int): Number of consecutive days for each year.
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
        # index: index of sites. To get one site's all years data
        x_seq = self.inputs[index, :, :]  # (years*days_of_year, num_input_features)
        y_seq = self.outputs[index, :, :]   # (years*days_of_year, num_output_features)
        # Convert to torch tensors.
        if not isinstance(x_seq, torch.Tensor):
            x_seq = torch.tensor(x_seq, dtype=torch.float32)
            y_seq = torch.tensor(y_seq, dtype=torch.float32)
        return x_seq, y_seq
    
# A commom class for Time series Models
class TimeSeriesModel_HiddenTransfer(TimeSeriesModel):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

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

    # To support transfer hidden state. Each site has multiple years data. Now needs transfer the hidden state between years
    # But, can't transfer the hidden state between sites.
    # When create batch for train and test, needs to make sure one site's all years data together
    def train_test_split_bysite(self, X, Y, num_sites, num_years, train_batch_size, days_per_year:int = 365, split_method = 'temporal', split_sequence_by_year=False):
        '''
        split_method: Train / Test dataset split method
        For example, the dataset includes 100 sites, and each site has 20 years data, N features

        'temporal': split by year. Choose the last two years as test dataset, train set is [100 sites, 18 years* days of year, N], test set is [100, 2* days of year, N]
        'spatial': split by site. Choose 20% sites as test set. Train set is [80, 20*days of year, N], test set is [20, 20*days of year, N]
        '''
        self.total_sites = num_sites
        self.total_years = num_years
        self.sequence_length = days_per_year

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
            shuffled_ix = torch.randperm(X.size()[0])
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
        train_dataset = SequenceDataset_multiYears(X_train, Y_train, days_per_year)
        test_dataset = SequenceDataset_multiYears(X_test, Y_test, days_per_year)

        # Create DataLoaders.
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"X_train shape is {X_train.shape}, Y_train shape {Y_train.shape}, X_test {X_test.shape}, Y_test {Y_test.shape}")

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

    # Train model with hidden state transfer
    def train_model(self, loss_fun, LR=0.001, step_size=20, gamma=0.8, maxepoch=80, use_y_mask=False):

        self.to(self.device)
        self.criterion = loss_fun # nn.MSELoss(), nn.L1Loss(), kgml_lib.multiTaskWeighted_Loss()
        self.train_y_mask = use_y_mask
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
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            train_predictions = []
            train_targets = []

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if use_y_mask:
                    y_mask = batch_y[..., self.output_dim:]
                    batch_y = batch_y[..., :self.output_dim] * y_mask

                optimizer.zero_grad()

                batch_loss = 0  # loss per batch
                hidden_state = None  # Initialize hidden state
                outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                batch_size = batch_x.shape[0]
                for _batch in range(batch_size): # one batch is one site's all years data
                    # Transfer hidden state between years, but can't between sites
                    for year in range(self.train_years):  # Loop over years
                        X_batch_year = batch_x[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)
                        Y_batch_year = batch_y[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)
                        if hidden_state is None:
                            y_pred, hidden_state = self(X_batch_year)
                        else:
                            y_pred, hidden_state = self(X_batch_year, hidden_state)
                        if use_y_mask:
                            y_pred = y_pred * y_mask
                        _loss = self.criterion(y_pred, Y_batch_year)
                        batch_loss += _loss
                        outputs_pred[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :] = y_pred[:,:,:]

                        if isinstance(hidden_state, tuple):  # LSTM
                            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                        else:  # GRU
                            hidden_state = hidden_state.detach()

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(batch_loss.item() / (self.train_years* batch_size))  # Normalize loss per year

                train_predictions.append(outputs_pred.cpu())
                train_targets.append(batch_y.cpu())
            
            train_predictions = torch.cat(train_predictions, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_R2 = compute_r2(train_predictions, train_targets)
            avg_train_loss = np.mean(train_losses)

            # Evaluate on the test set.
            self.eval()
            test_losses = []
            test_predictions = []
            test_targets = []
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    if use_y_mask:
                        y_mask = batch_y[..., self.output_dim:]
                        batch_y = batch_y[..., :self.output_dim] * y_mask

                    batch_loss = 0
                    hidden_state = None
                    outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                    batch_size = batch_x.shape[0]
                    for _batch in range(batch_size): # one batch is one site's all years data
                        for year in range(self.test_years):  
                            X_batch_year = batch_x[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)
                            Y_batch_year = batch_y[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)
                            if hidden_state is None:
                                y_pred, hidden_state = self(X_batch_year)
                            else:
                                y_pred, hidden_state = self(X_batch_year, hidden_state)
                            if use_y_mask:
                                y_pred = y_pred * y_mask
                            batch_loss += self.criterion(y_pred, Y_batch_year)
                            outputs_pred[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :] = y_pred[:,:,:]

                            if isinstance(hidden_state, tuple):  # LSTM
                                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                            else:  # GRU
                                hidden_state = hidden_state.detach()

                    test_losses.append(batch_loss.item() / (self.test_years * batch_size))

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
                # torch.save(self.state_dict(), checkpoint_path)
                torch.save(self.state_dict(), checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= 40:
                    print(f'\n#***# Early stopping triggered after {epoch+1} epochs!')
                    # self.load_state_dict(torch.load(checkpoint_path))
                    self.load_state_dict(torch.load(checkpoint_path))
                    break
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_test_loss)
            self.epochs = epoch+1
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            print(f'Train R2:  {train_R2.mean():.2f}', f'Test R2:  {test_R2.mean():.2f}')

    def test(self, use_y_mask=False):
        self.eval()
        self.test_y_mask = use_y_mask
        test_losses = []
        all_predictions = []  # Optional: to store predictions for further analysis
        all_targets = []      # Optional: to store true targets
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if use_y_mask:
                    y_mask = batch_y[..., self.output_dim:]
                    batch_y = batch_y[..., :self.output_dim] * y_mask
                
                outputs_pred = torch.zeros(batch_y.size(), device=self.device)
                total_loss = 0
                hidden_state = None # Initial hidden per batch
                batch_size = batch_x.shape[0]
                for _batch in range(batch_size): # one batch is one site's all years data
                    for year in range(self.test_years):  
                        X_batch_year = batch_x[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)
                        Y_batch_year = batch_y[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :].unsqueeze(0)

                        if hidden_state is None:
                            y_pred, hidden_state = self(X_batch_year)
                        else:
                            y_pred, hidden_state = self(X_batch_year, hidden_state)
                        
                        if use_y_mask:
                            y_pred = y_pred * y_mask
                        total_loss += self.criterion(y_pred, Y_batch_year)
                        outputs_pred[_batch, year*self.sequence_length:(year+1)*self.sequence_length, :] = y_pred[:,:,:]

                        if isinstance(hidden_state, tuple):  # LSTM
                            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                        else:  # GRU
                            hidden_state = hidden_state.detach()

                    test_losses.append(total_loss.item() / (self.test_years * batch_size))

                all_predictions.append(outputs_pred.cpu())
                all_targets.append(batch_y.cpu())
        
        # Calculate the average test loss
        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # concatenate all predictions and targets:
        self.all_predictions = torch.cat(all_predictions, dim=0)
        self.all_targets = torch.cat(all_targets, dim=0)
        compute_r2 = self.compute_r2
        test_R2 = compute_r2(self.all_predictions, self.all_targets)
        print(f'Test R2:  {test_R2.mean():.2f}')

    def check_results(self, device, check_xset, check_yset, num_years, use_y_mask):
        """
        check and record model output for later visualization (1 year only)
        check_x_set: the input that needs to check
        check_yset: the ground truth
        num_year: the number of year that need to record (from 0 to 17)
        """
        model_trained = self
        model_trained.to(device)
        model_trained.eval()

        if use_y_mask:
            y_mask = check_yset[..., self.output_dim:]
            check_yset = check_yset[..., :self.output_dim] * y_mask
        check_pred = torch.zeros(check_yset.size(), device = device)
        with torch.no_grad():
            for year in range(num_years):  
                X_batch_year = check_xset[:,year*self.sequence_length:(year+1)*self.sequence_length,:].to(device)  # shape: (batch, days, features)

                if year == 0:
                    outputs, hidden_state = model_trained(X_batch_year)  
                else:
                    outputs, hidden_state = model_trained(X_batch_year, hidden_state) 
                if use_y_mask:
                    outputs = outputs * y_mask
                check_pred[:,year*self.sequence_length:(year+1)*self.sequence_length,:] = outputs[:,:,:]

        return check_pred.cpu()


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

class GRUSeq2SeqWithAttention(TimeSeriesModel_HiddenTransfer):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, output_dim)  # Double input size due to concatenation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        if hidden == None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        gru_out, hidden = self.gru(x, hidden)  # gru_out shape: (batch_size, seq_len, hidden_dim)
        gru_out = self.dropout(gru_out)
        
        # Apply attention
        attended = self.attention(gru_out)  # (batch_size, seq_len, 2*hidden_dim)
        
        # Final projection
        out = self.fc(attended)  # (batch_size, seq_len, output_dim)
        return out, hidden
     
# A Simple GRU Regression model
class GRUSeq2Seq(TimeSeriesModel_HiddenTransfer):
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
class LSTMSeq2Seq(TimeSeriesModel_HiddenTransfer):
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
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_dim)
        if hidden == None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h,c)
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        out = self.fc(lstm_out)     # out shape: (batch_size, sequence_length, output_dim)
        return out, hidden


# the 2-cell N2O KGML model from Licheng's 2022 paper
class N2OGRU_KGML(TimeSeriesModel_HiddenTransfer):
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
class RecoGRU_KGML(TimeSeriesModel_HiddenTransfer):
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
    
