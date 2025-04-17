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



get_gpu_memory = kgml_lib.get_gpu_memory
n2o_my_loss = kgml_lib.gru_n2o_my_loss
my_loss = kgml_lib.my_loss
compute_r2 = kgml_lib.R2Loss()


class GRU:
    def __init__(self, input_path: str, output_path: str, input_data: str, output_model, dataset: Trad_ml_DataSet, sample_index_file: str = None, hidden_dim=64, num_layers=2):
        
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
        self.bsz = self.dataset.bsz
        self.mini_batch = 16
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_version = self.output_model
    
       
        self.X_train = self.dataset.X_train
        self.Y_train = self.dataset.Y_train
        self.X_val = self.dataset.X_val
        self.Y_val = self.dataset.Y_val


        train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.Y_train))
        val_dataset = TensorDataset(torch.Tensor(self.X_val), torch.Tensor(self.Y_val))
        train_loader = DataLoader(train_dataset, batch_size=self.mini_batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.mini_batch, shuffle=True, drop_last=True)


        self.train_loader = train_loader
        self.val_loader = val_loader
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.num_years = self.X_train.shape[1]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using:", self.device)



    def build_gru_model(self, input_dim, hidden_dim, output_dim, num_layers=2):
        "Build GRU model"
     
        class GRUNetwork(nn.Module):
            def __init__(self):
                super(GRUNetwork, self).__init__()
                self.num_layers = num_layers
                self.hidden_dim = hidden_dim
                self.gru_layer = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
                self.relu1 = nn.ReLU()
                self.fc1 = nn.Linear(hidden_dim, hidden_dim)
                self.relu2 = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, hidden_state=None):
                if hidden_state is None:
                    hidden_state = torch.zeros(self.num_layers, 
                                    x.size(0), self.hidden_dim).to(x.device)
                tuple_output = self.gru_layer(x, hidden_state) 
                output = tuple_output[0]
                hidden = tuple_output[1]
                output = self.relu1(output)   
                output = self.fc1(output)      
                output = self.relu2(output)    
                output = self.fc2(output)      
                return output, hidden
        
        return GRUNetwork()

    def train_step(self, learning_rate=0.001, num_epochs=500):
        print("Strat training")
        hidden_dim = self.hidden_dim
        num_layers = self.num_layers

        input_dim = self.X_train.size(3)
        output_dim = self.Y_train.size(3)
        device = self.device
       
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0] 

        # initialize model, loss function and optimizer
        model = self.build_gru_model(input_dim, hidden_dim, output_dim, num_layers=num_layers)
        print(model)
        model = nn.DataParallel(model)
        model.to(self.device)
        criterion = n2o_my_loss 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
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
                hidden_state = torch.zeros(num_layers, X_batch.size(0), hidden_dim).to(device)
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                total_loss = 0  
                total_loss1 = 0
                total_loss2 = 0
                total_loss3 = 0
                total_loss4 = 0
                total_loss5 = 0
                
                for year in range(X_batch.shape[1]):  
                    X_batch_year = X_batch[:, year, :, :]  
                    Y_batch_year = Y_batch[:, year, :, :]  
            

                    
                    if year == 0:
                        outputs, hidden_state = model(X_batch_year)
                        
                    else: 
                        hidden_state = torch.stack(hidden_state, dim=0)

                        outputs, hidden_state = model(X_batch_year, hidden_state)

               
                    loss, loss1, loss2, loss3, loss4, loss5 = criterion(outputs, Y_batch_year, loss_weights)
                    
             
                    total_loss += loss
                    total_loss1 += loss1
                    total_loss2 += loss2
                    total_loss3 += loss3
                    total_loss4 += loss4
                    total_loss5 += loss5

       
                    hidden_state = tuple(h.detach() for h in hidden_state)

  
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

      
                train_loss += total_loss.item() / self.num_years
                train_loss1 += total_loss1.item() / self.num_years
                train_loss2 += total_loss2.item() / self.num_years
                train_loss3 += total_loss3.item() / self.num_years
                train_loss4 += total_loss4.item() / self.num_years
                train_loss5 += total_loss5.item() / self.num_years
     
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
                    hidden_state = ()  
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    total_loss = 0  
                    total_loss1 = 0
                    total_loss2 = 0
                    total_loss3 = 0
                    total_loss4 = 0
                    total_loss5 = 0
                    for year in range(self.num_years): 
                        
                        X_batch_year = X_batch[:, year, :, :] 
                        Y_batch_year = Y_batch[:, year, :, :]  
                        if year == 0:
                            outputs, hidden_state = model(X_batch_year)
                        else: 
                            outputs, hidden_state = model(X_batch_year, hidden_state)
  
                        loss, loss1, loss2, loss3, loss4, loss5 = criterion(outputs, Y_batch_year, loss_weights)
                        total_loss += loss
                        total_loss1 += loss1
                        total_loss2 += loss2
                        total_loss3 += loss3
                        total_loss4 += loss4
                        total_loss5 += loss5
       
                        
                    val_loss += total_loss.item() / self.num_years
                    val_loss1 += total_loss1.item() / self.num_years
                    val_loss2 += total_loss2.item() / self.num_years
                    val_loss3 += total_loss3.item() / self.num_years
                    val_loss4 += total_loss4.item() / self.num_years
                    val_loss5 += total_loss5.item() / self.num_years
                


         
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

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            hidden_state = None
            for year in range(self.num_years):
                X_batch_year = check_xset[:, year, :, :].to(device)   # shape: (day, feature)
                Y_batch_year = check_yset[:, year, :, :].to(device)   # shape: (day, feature)

                if hidden_state is None:
                    outputs, hidden_state = model_trained(X_batch_year)
                else:
                    outputs, hidden_state = model_trained(X_batch_year, hidden_state)

                outputs = outputs.squeeze(1)  # shape: (day, feature)
                all_outputs.append(outputs.cpu())
                all_labels.append(Y_batch_year.cpu())

        # (year, day, feature)
        self.check_y_all_pred = torch.stack(all_outputs, dim=0).permute(1, 0, 2, 3)  # shape: (site, year, day, feature)
        self.check_y_all_true = torch.stack(all_labels, dim=0).permute(1, 0, 2, 3)  # shape: (site, year, day, feature)
        


        mtime = time.time()
        print("finished", f"Spending time: {mtime - starttime:.2f}s")



    def test(self):
        hidden_dim = self.hidden_dim
        num_layers = self.num_layers
        model_version = self.model_version

        device = self.device
        batch_size = self.mini_batch
        starttime = time.time()  
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0] 
        print("Testing started...")
        input_dim = self.X_train.size(3)
        output_dim = self.Y_train.size(3)

        # load pretrained model
        path_save = self.output_path + model_version
        checkpoint = torch.load(path_save, map_location=device)
        model = self.build_gru_model(input_dim, hidden_dim, output_dim, num_layers=num_layers) 
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        criterion = n2o_my_loss 
        model.to(device)

        
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
                total_loss = 0 
                total_loss1 = 0
                total_loss2 = 0
                total_loss3 = 0
                total_loss4 = 0
                total_loss5 = 0
                
                for year in range(self.num_years): 
                        X_batch_year = X_batch[:, year, :, :] 
                        Y_batch_year = Y_batch[:, year, :, :]  
                        if year == 0:
                            outputs, hidden_state = model(X_batch_year)
                        else: 
                            outputs, hidden_state = model(X_batch_year, hidden_state)
  
                        loss, loss1, loss2, loss3, loss4, loss5 = criterion(outputs, Y_batch_year, loss_weights)
                        total_loss += loss
                        total_loss1 += loss1
                        total_loss2 += loss2
                        total_loss3 += loss3
                        total_loss4 += loss4
                        total_loss5 += loss5
                        




          
                loss, loss1, loss2, loss3, loss4, loss5 = criterion(outputs, Y_batch_year, loss_weights)
                test_loss += total_loss.item() / self.num_years
                test_loss1 += total_loss1.item() / self.num_years
                test_loss2 += total_loss2.item() / self.num_years
                test_loss3 += total_loss3.item() / self.num_years
                test_loss4 += total_loss4.item() / self.num_years
                test_loss5 += total_loss5.item() / self.num_years

    
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
       
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        ncols = 1  
        nrows = 5 
 
        fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
        plot_names =  ["N2O_FLUX", "CO2_FLUX", "NO3_3", "NH4_3", "WTR_3"]
        test_names = ['Daily']  


        for i in range(len(plot_names)):  
            check_set = self.check_y_all_true[:, :, :, i].reshape(-1, self.check_y_all_true.shape[3]) # shape: (16, 18*365, :)
            check_set_pred = self.check_y_all_pred[:, :, :, i].reshape(-1, self.check_y_all_pred.shape[3]) # shape: (16, 18*365, :)
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
    data_path = './data/'
    input_data = 'input16_output5_pretrain_18yr.sav'
    output_path = "./gru_test_results/"
    sample_index_file = 'n2o_sample_index.sav'
    output_model = 'n2o_trad_ml.sav'
    scaler_file = 'input16_output5_scalers.sav'

    dataset = Trad_ml_DataSet(data_path=data_path, scaler_file=scaler_file,
                            input_data=input_data,
                            out_path=output_path)

    


    ml_model = GRU(input_path=data_path, output_path=output_path, 
                        input_data=input_data, output_model=output_model, 
                        dataset=dataset)
    
    dataset.print_dataset_info()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #ml_model.train_step()  
    ml_model.test('n2o_trad_ml.sav', device=device)
    ml_model.vis_loss()  
    ml_model.vis_flux()
    ml_model.vis_data_time_series()


  
