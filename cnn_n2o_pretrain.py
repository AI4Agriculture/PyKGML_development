import time
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import Trad_ml_DataSet
from torch.utils.data import TensorDataset, DataLoader
import kgml_lib
import matplotlib.pyplot as plt
import os
from kgml_lib import Z_norm_reverse
import pandas as pd



get_gpu_memory = kgml_lib.get_gpu_memory
n2o_my_loss = kgml_lib.gru_n2o_my_loss
my_loss = kgml_lib.my_loss
compute_r2 = kgml_lib.R2Loss()

class CNN:
    def __init__(self, input_path: str, output_path: str, input_data: str, output_model, dataset: Trad_ml_DataSet, sample_index_file: str = None, cnn_channels=[64, 32, 16], kernel_sizes=[7, 5, 3], hidden_dims=256, dropout=0.5):
        self.input_path = input_path
        self.output_path = output_path
        self.input_data = self.input_path + input_data
        self.path_save = output_path + output_model
        self.sample_index_file = sample_index_file
        self.dataset = dataset
        self.train_losses = []
        self.val_losses = []
        self.output_model = output_model
        
       
        self.dataset.load_data()        
        self.dataset.parse_data_with_window()    
        self.dataset.train_test_split()

        ## set to 16 because the test set only has 16 batch in total
        self.mini_batch = 256
        self.kernel_sizes = kernel_sizes
        self.cnn_channels = cnn_channels
        self.hidden_dims = hidden_dims
        self.dropout = dropout
    
       
        self.X_train = self.dataset.X_train.reshape(-1, 365, self.dataset.n_X)
        self.Y_train = self.dataset.Y_train.reshape(-1, 365, self.dataset.n_Y)
        self.X_val = self.dataset.X_val.reshape(-1, 365, self.dataset.n_X)
        self.Y_val = self.dataset.Y_val.reshape(-1, 365, self.dataset.n_Y)
  


        train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.Y_train))
        val_dataset = TensorDataset(torch.Tensor(self.X_val), torch.Tensor(self.Y_val))
        train_loader = DataLoader(train_dataset, batch_size=self.mini_batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.mini_batch, shuffle=True, drop_last=True)


        self.train_loader = train_loader
        self.val_loader = val_loader
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using:", self.device)



    def build_model(self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len=365,
        cnn_channels=[64, 32, 16],
        kernel_sizes=[7, 5, 3],
        dropout=0.3
    ):
        class CNN_Model(nn.Module):
            def __init__(self, input_dim, output_dim, seq_len, cnn_channels, kernel_sizes, hidden_dim, dropout):
                super(CNN_Model, self).__init__()
                self.seq_len = seq_len
                self.output_dim = output_dim

                assert len(cnn_channels) == len(kernel_sizes), "cnn_channels and kernel_sizes must match in length"

                layers = []
                in_channels = input_dim
                for out_channels, k in zip(cnn_channels, kernel_sizes):
                    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    in_channels = out_channels
                self.conv_net = nn.Sequential(*layers)

                conv_out_dim = int(cnn_channels[-1]) * int(seq_len)
                self.flatten = nn.Flatten()


                self.mlp = nn.Sequential(
                    nn.Linear(conv_out_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim * seq_len)
                )

            def forward(self, x):
               
                x = x.permute(0, 2, 1)
                x = self.conv_net(x)
                x = self.flatten(x)
                x = self.mlp(x)
                x = x.view(-1, self.seq_len, self.output_dim)
                return x

        return CNN_Model(input_dim, output_dim, seq_len, cnn_channels, kernel_sizes, hidden_dim, dropout)


    def train_step(self, learning_rate=0.001, num_epochs=500): 
        device = self.device
        input_dim = self.X_train.size(2) 
        output_dim = self.Y_train.size(2) 
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        hidden_dim = self.hidden_dims
        kernel_sizes = self.kernel_sizes
        dropout = self.dropout
        cnn_channels = self.cnn_channels
        seq_len = 365

  
        model = self.build_model(input_dim, hidden_dim, output_dim, seq_len, cnn_channels, kernel_sizes, dropout)
        model = nn.DataParallel(model)
        model.to(device)
        print(model)

        criterion = n2o_my_loss  
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
       
        train_losses = []
        val_losses = []
        starttime = time.time() 

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_loss1 = 0.0
            train_loss2 = 0.0
            train_loss3 = 0.0
            train_loss4 = 0.0
            train_loss5 = 0.0
            
         
            for X_batch, Y_batch in self.train_loader:
                optimizer.zero_grad()
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
 
            
             
                outputs = model(X_batch)
                total_loss, total_loss1, total_loss2, total_loss3, total_loss4, total_loss5 = criterion(outputs, Y_batch, loss_weights)
              
                total_loss.backward()
                optimizer.step()

      
                train_loss += total_loss.item() 
                train_loss1 += total_loss1.item()
                train_loss2 += total_loss2.item()
                train_loss3 += total_loss3.item() 
                train_loss4 += total_loss4.item()
                train_loss5 += total_loss5.item() 
     
            scheduler.step()
        
            
            model.eval()
            val_loss = 0.0
            val_loss1 = 0.0
            val_loss2 = 0.0
            val_loss3 = 0.0
            val_loss4 = 0.0
            val_loss5 = 0.0
            with torch.no_grad():
                
                for X_batch, Y_batch in self.val_loader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    outputs = model(X_batch)
 
                        

                    total_loss, total_loss1, total_loss2, total_loss3, total_loss4, total_loss5 = criterion(outputs, Y_batch, loss_weights)
                        
                        
                    val_loss += total_loss.item()
                    val_loss1 += total_loss1.item() 
                    val_loss2 += total_loss2.item() 
                    val_loss3 += total_loss3.item()
                    val_loss4 += total_loss4.item() 
                    val_loss5 += total_loss5.item()
                
            train_losses.append([train_loss / len(self.train_loader), train_loss1 / len(self.train_loader), 
                                train_loss2 / len(self.train_loader), train_loss3 / len(self.train_loader),
                                train_loss4 / len(self.train_loader), train_loss5 / len(self.train_loader)])
            
            val_losses.append([val_loss / len(self.val_loader), val_loss1 / len(self.val_loader), 
                            val_loss2 / len(self.val_loader), val_loss3 / len(self.val_loader),
                            val_loss4 / len(self.val_loader), val_loss5 / len(self.val_loader)])

         
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(self.train_loader):.4f}, "
                f"Val Loss: {val_loss / len(self.val_loader):.4f}, "
                f"Train Loss Components: [{train_loss1 / len(self.train_loader):.4f}, "
                f"{train_loss2 / len(self.train_loader):.4f}, "
                f"{train_loss3 / len(self.train_loader):.4f}, "
                f"{train_loss4 / len(self.train_loader):.4f}, "
                f"{train_loss5 / len(self.train_loader):.4f}]")
            
            ckpt_dir = os.path.join(self.output_path, "ckpt/")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses
            }, ckpt_dir + f'model_epoch_{epoch}.pth')

        endtime = time.time() 
        print(f"Training completed in {endtime - starttime:.2f} seconds.")
        
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, self.path_save)




    def check_results(self, device, model_trained, check_xset, check_yset, num_year, starttime):
        """
        Check and record model output for later visualization (all years).
        The output will be saved as tensors of shape (year, day, feature).
        """
        model_trained.to(device)
        model_trained.eval()


        with torch.no_grad():


            X_batch = check_xset
            Y_batch = check_yset

            outputs = model_trained(X_batch)
        
       



        self.check_y_all_pred = outputs.cpu() 
        self.check_y_all_true = Y_batch.cpu()  


        mtime = time.time()
        print("finished", f"Spending time: {mtime - starttime:.2f}s")

    def test(self):
        
        starttime = time.time()  
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0] 
        print("Testing started...")
        input_dim = self.X_train.size(2)
        output_dim = self.Y_train.size(2)
        hidden_dim = self.hidden_dims
        kernel_sizes = self.kernel_sizes
        dropout = self.dropout
        cnn_channels = self.cnn_channels
        seq_len = 365

        ## load the last checkpoint
        ckpt_dir = os.path.join(self.output_path, "ckpt/")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        checkpoints = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pth")],
                key=lambda x: int(x.split("_")[-1].split(".")[0]) 
            )
        if not checkpoints:
                print("No checkpoint files found!")
                return
        device = self.device
        latest_ckpt = checkpoints[-1] 
        path_save = os.path.join(ckpt_dir, latest_ckpt)
        checkpoint = torch.load(path_save, map_location=device)
   

  
        model = self.build_model(input_dim, hidden_dim, output_dim, seq_len, cnn_channels, kernel_sizes, dropout)
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        criterion = n2o_my_loss  
 
        val_loader = self.val_loader
        test_loss = 0.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        test_loss4 = 0.0
        test_loss5 = 0.0

    
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                


                
                outputs = model(X_batch)

                        

                total_loss, total_loss1, total_loss2, total_loss3, total_loss4, total_loss5 = criterion(outputs, Y_batch, loss_weights)
                        
    
          

                test_loss += total_loss.item() 
                test_loss1 += total_loss1.item()
                test_loss2 += total_loss2.item() 
                test_loss3 += total_loss3.item() 
                test_loss4 += total_loss4.item() 
                test_loss5 += total_loss5.item() 


        check_xset = self.X_val  # shape: (16 ,18, days, features)
        check_yset = self.Y_val  # shape: (16 ,18, days, features)

        self.check_results(device, model, check_xset, check_yset, 1, starttime)


        endtime = time.time()
        print(f"Testing completed in {endtime - starttime:.2f} seconds.")
        print(f"Test Loss: {test_loss / len(val_loader):.4f}")
        print(f"Test Loss Components: [{test_loss1 / len(val_loader):.4f}, "
            f"{test_loss2 / len(val_loader):.4f}, "
            f"{test_loss3 / len(val_loader):.4f}, "
            f"{test_loss4 / len(val_loader):.4f}, "
            f"{test_loss5 / len(val_loader):.4f}]")


    def vis_loss(self):
        plt.rcParams.update({'font.size': 20})
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

        fig, ax = plt.subplots(6, 1, figsize=(12, 18))  
        lw = 3.0

        plot_names = ["All", "N2O_FLUX", "CO2_FLUX", "NO3_3", "NH4_3", "WTR_3"]

        
        ckpt_dir = os.path.join(self.output_path, "ckpt/")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        checkpoints = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]) 
        )


        if not checkpoints:
            print("No checkpoint files found!")
            return
        latest_ckpt = checkpoints[-1]

        path_save = os.path.join(ckpt_dir, latest_ckpt)
  

        try:
            checkpoint = torch.load(path_save, map_location=torch.device('cpu'))
            train_losses = np.array(checkpoint['train_loss'])
            val_losses = np.array(checkpoint['val_loss'])
        except Exception as e:
            print(f"Error loading model {latest_ckpt}: {e}")
            return

        for i in range(len(plot_names)):
            ax[i].plot(train_losses[:, i], label=f"Train loss: {plot_names[i]}", lw=lw)
            ax[i].plot(val_losses[:, i], label=f"Val loss: {plot_names[i]}", lw=lw, linestyle="--")

            ax[i].set_ylabel('Loss', fontweight='bold')
            ax[i].set_xlabel("Epoch")
            ax[i].legend(loc='upper right')
            ax[i].set_title(f'{plot_names[i]} Loss')

        plt.tight_layout()

        print("Loss vis saved")
        plt.savefig(self.output_path + 'training_and_val_loss.png', dpi=300, bbox_inches='tight')
        plt.show()

    def vis_flux(self):
        ########################################################################################
        ### Visualize flux prediction vs synthetic ground truth
        ########################################################################################
        data = self.dataset
        total_b = self.X_val.size(0)
        origin_shape = self.Y_val.shape
       
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        ncols = 1  
        nrows = 5 
 
        fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
        plot_names =  ["N2O_FLUX", "CO2_FLUX", "NO3_3", "NH4_3", "WTR_3"]
        test_names = ['Daily']  


        for i in range(len(plot_names)):  
            check_set = self.check_y_all_true.reshape(-1, origin_shape[-1]) # shape: (16, 18*365, :)
            check_set_pred = self.check_y_all_pred.reshape(-1, origin_shape[-1]) # shape: (16, 18*365, :)

            # check_set = check_set.reshape(data.Y_val.shape[2] * data.Y_val.shape[1], data.Y_val.shape[3])
            Ysim = Z_norm_reverse(check_set_pred[:, i], data.scaler[16+i, :], 1.0).to("cpu")
            Yobs = Z_norm_reverse(check_set[:, i], data.scaler[16+i, :], 1.0).to("cpu")


            ax[i].scatter(
                Yobs.contiguous().view(-1).cpu().numpy(),  
                Ysim.contiguous().view(-1).cpu().numpy(),  
                s=10, color='black', alpha=0.5
            )       

            R2 = compute_r2(Ysim, Yobs).numpy()
            RMSE = np.sqrt(my_loss(Ysim, Yobs).numpy())
            Bias = torch.mean(Ysim - Yobs).numpy()
  
            m, b, r_value, p_value, std_err = stats.linregress(Yobs.contiguous().view(-1).numpy(),Ysim.contiguous().view(-1).numpy())

            lim_min = min(torch.min(Ysim).numpy(), torch.min(Yobs).numpy())
            lim_max = max(torch.max(Ysim).numpy(), torch.max(Yobs).numpy())
            ax[i].set_xlim([lim_min, lim_max])
            ax[i].set_ylim([lim_min, lim_max])
            ax[i].text(lim_min + 0.1 * (lim_max - lim_min), lim_max - 0.1 * (lim_max - lim_min),
                    'R$^2$=%0.3f\nRMSE=%0.3f\nBias=%0.3f' % (R2, RMSE, Bias), fontsize=12, ha='left', va='top')
            ax[i].set_xlabel('Ecosys simulated', fontsize=15, weight='bold')
            ax[i].set_ylabel('KGML predicted ' + '(g C $m^{-2}$ $day^{-1}$)', fontsize=15, weight='bold')
            ax[i].set_title(plot_names[i] + ' (' + test_names[0] + ')', fontsize=15, weight='bold')
            ax[i].plot(Yobs, m * Yobs + b, color='steelblue', lw=1.0)
            ax[i].plot([lim_min, lim_max], [lim_min, lim_max], color='red', linestyle='--')
   
        print("Flux vis saved")
        plt.savefig(self.output_path + 'test_flux_prediction_vs_ground_truth.png', dpi=300, bbox_inches='tight')
        plt.show()

    def vis_data_time_series(self):
        data = self.dataset
        origin_shape = data.Y_val.shape
 
        check_set = self.check_y_all_true.reshape(origin_shape[0], -1, origin_shape[-1]) # shape: (16, 18*365, :)
        check_set_pred = self.check_y_all_pred.reshape(origin_shape[0], -1, origin_shape[-1]) # shape: (16, 18*365, :)
        x_feature = self.X_val.reshape(origin_shape[0], -1, 16) # shape: (16, 18*365, :)
        
        sample_idx = 0  
        plot_names = ["N2O_FLUX", "CO2_FLUX", "NO3_3", "NH4_3", "WTR_3"]
        plot_x_indices = ['FERTZR_N','RADN','TMAX_AIR','TMIN_AIR','HMAX_AIR','HMIN_AIR','WIND','PRECN']
        time_steps = check_set_pred.shape[1]


        fig, ax = plt.subplots(len(plot_names) + len(plot_x_indices), 1, figsize=(15, 2.5 * (len(plot_names) + len(plot_x_indices))))
        lw = 2.5
        plt.subplots_adjust(hspace=10)

        for i in range(len(plot_names)):
           

            Y_obs = Z_norm_reverse(check_set[sample_idx, :, i], data.scaler[16+i, :], 1.0).to("cpu")
            Y_sim = Z_norm_reverse(check_set_pred[sample_idx, :, i], data.scaler[16+i, :], 1.0).to("cpu")


            ax[i].plot(np.arange(time_steps), Y_obs, lw=lw, color='blue', label='Observed')
            ax[i].plot(np.arange(time_steps), Y_sim, lw=lw, color='red', linestyle='--', label='Predicted')
            ax[i].set_ylabel(plot_names[i])
            ax[i].set_xlabel("Time (days)")
            # ax[i].set_title(f"{plot_names[i]}")

        for j in range(len(plot_x_indices)):
            X_ts = Z_norm_reverse(x_feature[sample_idx, :, j], data.scaler[j, :], 1.0).to("cpu")
            ax[len(plot_names) + j].plot(np.arange(time_steps), X_ts, lw=lw, color='green')
            ax[len(plot_names) + j].set_ylabel(plot_x_indices[j])
            ax[len(plot_names) + j].set_xlabel("Time (days)")
            # ax[len(plot_names) + j].set_title(f"Input Feature: {plot_x_indices[j]}")

        plt.tight_layout()
        plt.savefig(self.output_path + "Time_series.png", dpi=300)
        plt.show()
        print("Time series vis saved")




if __name__  == "__main__":
    data_path = './data'
    input_data = 'input16_output5_pretrain_18yr.sav'
    scaler_file = 'input16_output5_scalers.sav'
    output_path = "./test_results_cnn/"
    sample_index_file = 'n2o_sample_index.sav'
    output_model = 'n2o_trad_ml.sav'

    # optimization parameters
    learning_rate = 0.001
    num_epochs = 50

    # model parameters
    hidden_dim = 128
   
    kernel_sizes = [7, 5, 3]
    cnn_channels = [64, 32, 16]
    hidden_dims = 256
    dropout = 0.5


    dataset = Trad_ml_DataSet(data_path=data_path, scaler_file=scaler_file,
                            input_data=input_data,
                            out_path=output_path)

    


    ml_model = CNN(input_path=data_path, output_path=output_path, 
                        input_data=input_data, output_model=output_model, 
                        dataset=dataset, kernel_sizes=kernel_sizes,
                        cnn_channels=cnn_channels, hidden_dims=hidden_dims, dropout=dropout)
    
    dataset.print_dataset_info()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ml_model.train_step(learning_rate=learning_rate, num_epochs=num_epochs)  
    ml_model.test()
    ml_model.vis_loss()  
    ml_model.vis_flux()
    ml_model.vis_data_time_series()
