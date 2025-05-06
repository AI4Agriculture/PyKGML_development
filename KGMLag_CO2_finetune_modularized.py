import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import os
from io import open
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import kgml_lib

from dataset import Step5_DataSet

Z_norm = kgml_lib.Z_norm 
Z_norm_reverse = kgml_lib.Z_norm_reverse
Z_norm_with_scaler = kgml_lib.Z_norm_with_scaler
get_gpu_memory = kgml_lib.get_gpu_memory
compute_r2=kgml_lib.R2Loss()
myloss_mb_flux_mask_re_v2 = kgml_lib.myloss_mb_flux_mask_re_v2
myloss_mb_flux_mask_re =kgml_lib. myloss_mb_flux_mask_re
check_Ra_Rh_response_v2 = kgml_lib.check_Ra_Rh_response_v2
Get_Ra_Rh_org = kgml_lib.Get_Ra_Rh_org
check_Rh_response_v2 = kgml_lib.check_Rh_response_v2

class Kgml_Model:
    def __init__(self, input_path:str, output_path:str, pretrained_model:str, output_model:str, 
                 synthetic_data:str, dataset:Step5_DataSet):
        self.input_path = input_path
        self.output_path = output_path
        self.pretrained_model = input_path + pretrained_model
        self.output_model = output_model
        self.path_save = output_path + output_model
        self.dataset = dataset

        self.synthetic_data = synthetic_data

        self.rangess = [[-2, 2],[-3.5, 2.5],[-2.5, 5],[-1.5,3.5],[-1.5,7],[-2.0,5.5],[-0.3889,10.0],[-1,1],[-0.5,5],[-1.6384,1.6384],\
          [-10,2],[-0.75,4],[-3.25,1.25],[-4.5,5.0],[-2.5,3],[-0.75,7.75],[-0.5,5],[-3.25,4.25],[-1.75,3.5]]
        self.intervals = [0.2,0.3,0.5,0.25,0.5,0.5,0.5,2,0.25,1.6384*2/17,0.5,0.25,0.25,0.5,0.25,0.5,0.25,0.5,0.25]
        #########test different interval multiplier
        self.intv_f = 3.0
        ########set tolerance r for each variable
        self.Ra_r_mins = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.Rh_r_mins = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.99, 0.9, 0.9]

        # Check if GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available. Initialization failed.")
        
        # Proceed with GPU setup
        self.device = torch.device("cuda")
        print("GPU is available. Using:", self.device)

    def details(self):
        model = self.model
        print(model)
        params = list(model.parameters())
        print(len(params))
        print(params[5].size())  # conv1's .weight
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    def load_synthetic_data(self):
        full_name = self.input_path + self.synthetic_data
        if os.path.exists(full_name):
            tmp = torch.load(full_name, weights_only=False) #'sys_data2.sav'
            key = self.synthetic_data.split('.')[0]
            self.check_xset = tmp[key] #tmp['sys_data2']
        else:
            raise FileNotFoundError(f"Synthetic_data {full_name} not exist.")

    def freeze_yield(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            if param.requires_grad and ('attn' in name or 'gru_basic' in name or 'densor_yield' in name):
                param.requires_grad = False
            if param.requires_grad:
                print(name)

    def load_RecoGRU_multitask_v11_3(self, drop_out:float = 0.2, seq_len:int = 365):
        self.seq_len= seq_len
        data = self.dataset
        n_f = data.n_f # number of features

        GPP_index = data.fts_names_1.index('GPP') #8
        GPP_scaler = data.X_scaler[GPP_index,:]

        Ra_index = data.outNames_1.index('Ra') #0
        Ra_scaler = data.Y1_scaler[Ra_index,:] #[0,:]

        Yield_scaler = data.Y2_scaler[0,:]
        Res_scaler = data.Res_scaler[0,:]

        checkpoint=torch.load(self.pretrained_model, weights_only=False)
        self.model= kgml_lib.RecoGRU_multitask_v11_3(n_f,drop_out,seq_len,GPP_scaler,
                                                Ra_scaler,Yield_scaler,Res_scaler)

        ##load the pre-trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_Ra_Rh_daily(self):
        Ra_daily_orgs =[]
        Rh_daily_orgs =[]
        for i in range(4):
            for j in range(5):
                if not (i==3 and j ==4):
                    feature = i*5+j
                    ranges = self.rangess[feature]
                    #########test different interval multiplier
                    if feature == 7:
                        interval = self.intervals[feature]
                    else:
                        interval = self.intervals[feature]*self.intv_f
                    
                    with torch.no_grad():
                        Ra_daily_org, Rh_daily_org = Get_Ra_Rh_org(self.model,self.check_xset,feature, ranges, interval,
                                                                   self.dataset.Y1_scaler, self.device)
                        Ra_daily_orgs.append(Ra_daily_org)
                        Rh_daily_orgs.append(Rh_daily_org)

        self.Ra_daily_orgs = Ra_daily_orgs
        self.Rh_daily_orgs = Rh_daily_orgs

    def train_step5(self, LR=0.001,step_size=30, gamma=0.6, maxepoch=30*4):
        
        #self.load_synthetic_data()
        self.freeze_yield()
        data = self.dataset

        device = self.device
        model1 = self.model

        #self.get_Ra_Rh_daily()

        #initials
        lamda = [1.0,1.0]
        tol_mb = 0.01
        loss_val_best = 500000
        R2_best=0.5
        best_epoch = 1500
        #train the model and pring loss/ yearly training
        lr_adam=LR #orginal 0.0001
        optimizer = optim.Adam([
                    {'params': model1.gru_NEE.parameters(), 'lr': lr_adam*0.5},
                    {'params': model1.densor_NEE.parameters(), 'lr': lr_adam*0.5},
                    {'params': model1.gru_Rh.parameters(), 'lr': lr_adam*0.2},
                    {'params': model1.densor_Rh.parameters(), 'lr': lr_adam*0.2},
                    {'params': model1.gru_Ra.parameters(), 'lr': lr_adam*0.5},
                    {'params': model1.densor_Ra.parameters(), 'lr': lr_adam*0.5},
                    {'params': model1.attn.parameters()},
                    {'params': model1.densor_yield.parameters()},
                    {'params': model1.gru_basic.parameters()}
                ], lr=lr_adam) #add weight decay normally 1-9e-4
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        train_n = len(data.X_sites) # 11 sites
        #during training
        slw=365
        slw05=365
        train_losses = []
        val_losses = []
        #maxepoch=maxepoch
        starttime=time.time()
        model1.train()
        for epoch in range(maxepoch):
            train_loss=0.0
            train_loss1=0.0
            train_loss2=0.0
            train_loss3=0.0
            train_loss4=0.0
            val_loss=0.0
            #shuffled the training data
            shuffled_b=torch.randperm(train_n)
            Y_Reco_obs_new = []
            Y_NEE_obs_new = []
            Y_masks_train_new = []
            X_sites_new = []
            for i in range(train_n):
                Y_Reco_obs_new.append(data.Y_Reco_obs[shuffled_b[i]])
                Y_NEE_obs_new.append(data.Y_NEE_obs[shuffled_b[i]])
                Y_masks_train_new.append(data.Y_masks_train[shuffled_b[i]])
                X_sites_new.append(data.X_sites[shuffled_b[i]])
            
            Y_Ra_pred_all=[]
            Y_Rh_pred_all=[]
            Y_Reco_pred_all=[]
            Y_NEE_pred_all=[]
            model1.zero_grad()
            ###keep Rh responses
            if epoch > 0:
                for feature in range(len(self.rangess)):
                    if feature == 7: # 'Crop_Type'
                        interval = self.intervals[feature]
                    else:
                        interval = self.intervals[feature]*self.intv_f
                    loss4 = check_Ra_Rh_response_v2(model1, self.check_xset, feature, self.rangess[feature], interval,
                                                    self.Ra_r_mins[feature], self.Ra_daily_orgs[feature],
                                                    self.Rh_r_mins[feature], self.Rh_daily_orgs[feature],
                                                    data.Y1_scaler, device)
                    optimizer.zero_grad()
                    loss4.backward()
                    optimizer.step()  
                    with torch.no_grad():
                        train_loss4 += loss4.item()
            
            for bb in range(train_n):
                totsq = X_sites_new[bb].size(1) # years * 365
                maxit = int((totsq-slw)/slw05+1) # years
                #model initials
                with torch.no_grad():
                    hidden = model1.init_hidden(1)
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(X_sites_new[bb][:,slw05*it:slw05*it+slw,:],hidden)
                Y1_pred_sq = torch.zeros([1,totsq,3],device=device)
                for it in range(maxit):
                    #print(sbb,ebb,X_train_new[slw05*it:slw05*it+slw,sbb:ebb,:].size(),hidden.size(),X_train_new.size())
                    Y1_pred,Y2_pred,hidden = model1(X_sites_new[bb][:,slw05*it:slw05*it+slw,:],hidden)
                    Y1_pred_sq[:,slw05*it:slw05*it+slw,:] = Y1_pred[:,:,:]
                ####separate Ra and Rh
                Ra_temp = -1.0*Z_norm_reverse(Y1_pred_sq[0,:,0],data.Y1_scaler[0,:],1.0).view(-1)
                Rh_temp = -1.0*Z_norm_reverse(Y1_pred_sq[0,:,1],data.Y1_scaler[1,:],1.0).view(-1)
                Y_Ra_pred_all.append(Ra_temp)
                Y_Rh_pred_all.append(Rh_temp)
                Y_Reco_pred_all.append(Ra_temp+Rh_temp)
                Y_NEE_pred_all.append(Z_norm_reverse(Y1_pred_sq[0,:,2],data.Y1_scaler[2,:],1.0).view(-1))
                for zz in range(len(hidden)):
                    hidden[zz].detach_()  
            #######MSE and mass balance loss
            loss,loss1,loss2,loss3 = myloss_mb_flux_mask_re_v2(Y_Ra_pred_all,Y_Rh_pred_all,Y_NEE_pred_all,
                                                    Y_Reco_obs_new, Y_NEE_obs_new,
                                    X_sites_new,data.X_scaler[8,:], data.Y1_scaler,
                                    Y_masks_train_new, lamda,tol_mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss=train_loss+loss.item()
                train_loss1=train_loss1+loss1.item()
                train_loss2=train_loss2+loss2.item()
                train_loss3=train_loss3+loss3.item()
            scheduler.step()
            
            #validation
            model1.eval()
            with torch.no_grad():
                train_loss=[train_loss,train_loss1/train_n,train_loss2/train_n,train_loss3/train_n,train_loss4/data.n_f]
                train_losses.append(train_loss)
                #R2 for all
                Y_masksb =[]
                for i in range(train_n):
                    Y_masksb.append(Y_masks_train_new[i].ge(0.5))
                Y_masksb = torch.cat(Y_masksb,dim=0)
                
                Reco_pred_masked = torch.masked_select(torch.cat(Y_Reco_pred_all,dim=0),Y_masksb)
                Reco_obs_masked = torch.masked_select(torch.cat(Y_Reco_obs_new,dim=0),Y_masksb)
                NEE_pred_masked = torch.masked_select(torch.cat(Y_NEE_pred_all,dim=0),Y_masksb)
                NEE_obs_masked = torch.masked_select(torch.cat(Y_NEE_obs_new,dim=0),Y_masksb)
                train_R2 = [compute_r2(Reco_pred_masked.contiguous().view(-1),
                            Reco_obs_masked.contiguous().view(-1)).item(),
                        compute_r2(NEE_pred_masked.contiguous().view(-1),
                            NEE_obs_masked.contiguous().view(-1)).item()]

                Y_Ra_pred_all=[]
                Y_Rh_pred_all=[]
                Y_Reco_pred_all=[]
                Y_NEE_pred_all=[]
                for bb in range(train_n):
                    totsq = data.X_sites[bb].size(1)
                    maxit = int((totsq-slw)/slw05+1)
                    hidden = model1.init_hidden(1)
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(data.X_sites[bb][:,slw05*it:slw05*it+slw,:].to(device),hidden)
                    Y1_pred_sq = torch.zeros([1,totsq,3],device=device)
                    for it in range(maxit):
                        #change intitial for year round simulation
                        Y1_pred,Y2_pred, hidden = model1(data.X_sites[bb][:,slw05*it:slw05*it+slw,:].to(device),hidden)
                        Y1_pred_sq[:,slw05*it:slw05*it+slw,:] = Y1_pred[:,:,:]
                    Y_Reco_pred_all.append(-1.0*(Z_norm_reverse(Y1_pred_sq[0,:,0],
                                    data.Y1_scaler[0,:],1.0)+Z_norm_reverse(Y1_pred_sq[0,:,1],
                                    data.Y1_scaler[1,:],1.0)).view(-1))
                    Y_NEE_pred_all.append(Z_norm_reverse(Y1_pred_sq[0,:,2],data.Y1_scaler[2,:],1.0).view(-1))
                
                loss,loss1,loss2,loss3 = myloss_mb_flux_mask_re(Y_Reco_pred_all,Y_NEE_pred_all,data.Y_Reco_obs, data.Y_NEE_obs,
                                    data.X_sites, data.X_scaler[8,:], data.Y1_scaler,
                                    data.Y_masks_val, lamda,tol_mb)
                #calculate response to SOC:
                loss4 = 0.0
                for feature in range(len(self.rangess)):
                    if feature == 7:
                        interval = self.intervals[feature]
                    else:
                        interval = self.intervals[feature]*self.intv_f
                    loss4_t = check_Ra_Rh_response_v2(model1, self.check_xset, feature, self.rangess[feature], interval,
                                                            self.Ra_r_mins[feature], self.Ra_daily_orgs[feature],
                                                            self.Rh_r_mins[feature], self.Rh_daily_orgs[feature],
                                                            data.Y1_scaler, device)
                    loss4 = loss4+loss4_t
                val_loss=[loss.item(),loss1.item()/train_n,loss2.item()/train_n,loss3.item()/train_n,loss4.item()/data.n_f]
                val_losses.append(val_loss)
                #R2 for all
                Y_masksb =[]
                for i in range(train_n):
                    Y_masksb.append(data.Y_masks_val[i].ge(0.5))
                Y_masksb = torch.cat(Y_masksb,dim=0)
                
                Reco_pred_masked = torch.masked_select(torch.cat(Y_Reco_pred_all,dim=0),Y_masksb)
                Reco_obs_masked = torch.masked_select(torch.cat(data.Y_Reco_obs,dim=0),Y_masksb)
                NEE_pred_masked = torch.masked_select(torch.cat(Y_NEE_pred_all,dim=0),Y_masksb)
                NEE_obs_masked = torch.masked_select(torch.cat(data.Y_NEE_obs,dim=0),Y_masksb)
                val_R2 = [compute_r2(Reco_pred_masked.contiguous().view(-1),
                            Reco_obs_masked.contiguous().view(-1)).item(),
                        compute_r2(NEE_pred_masked.contiguous().view(-1),
                            NEE_obs_masked.contiguous().view(-1)).item()]
                #if val_loss < loss_val_best and val_R2 > R2_best:
                if val_loss[0] < loss_val_best:
                    loss_val_best=val_loss[0]
                    R2_best = val_R2
                    best_epoch = epoch
                    f0=open(self.path_save,'w')
                    f0.close()
                    #os.remove(path_save)
                    torch.save({'epoch': epoch,
                            'model_state_dict': model1.state_dict(),
                            'R2': train_R2,
                            'loss': train_loss,
                            'los_val': val_loss,
                            'R2_val': val_R2,
                            }, self.path_save)    
                print("finished training epoch", epoch+1)
                mtime=time.time()
                print("train_loss: ", train_loss, "train_R2", train_R2,"val_loss:",val_loss,"val_R2", val_R2,
                    "loss val best:",loss_val_best,"R2 val best:",R2_best, f"Spending time: {mtime - starttime}s")

                if np.mean(train_R2)> 0.999 or (epoch - best_epoch) > 30:
                    break
            model1.train()
        endtime=time.time()
        path_fs = self.path_save+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model1.state_dict(),
                    }, path_fs)  
        print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best)
        print(f"total Training time: {endtime - starttime}s")

    def vis_Ra_Rh(self):
        plt.rcParams.update({'font.size': 20})
        plt.rcParams['xtick.labelsize']=20
        plt.rcParams['ytick.labelsize']=20
        ncols = 5
        nrows = 4
        fig, ax = plt.subplots(nrows,ncols,figsize=(6*ncols, 5*nrows))
        fig1, ax1 = plt.subplots(nrows,ncols,figsize=(6*ncols, 5*nrows))
        # Ra_daily_orgs =[]
        # Rh_daily_orgs =[]
        for i in range(4):
            for j in range(5):
                if not (i==3 and j ==4):
                    feature = i*5+j
                    ranges = self.rangess[feature]
                    #########test different interval multiplier
                    if feature == 7:
                        interval = self.intervals[feature]
                    else:
                        interval = self.intervals[feature]*self.intv_f
                    total_n = int((ranges[1] - ranges[0])/interval)+1
                    # with torch.no_grad():
                    #     Ra_daily_org, Rh_daily_org = Get_Ra_Rh_org(model,self.check_xset,feature, ranges, interval,
                    #                                                self.dataset.Y1_scaler, self.device)
                    #     Ra_daily_orgs.append(Ra_daily_org)
                    #     Rh_daily_orgs.append(Rh_daily_org)

                    Ra_daily_org = self.Ra_daily_orgs[feature]
                    Rh_daily_org = self.Rh_daily_orgs[feature]
                    x = np.zeros([total_n])
                    for zz in range(total_n):
                        x[zz] = ranges[0] + interval*zz
                    #reverse value to real one:
                    feature_name = self.dataset.f_names[feature]
                    x = Z_norm_reverse(x,self.dataset.X_scaler[feature,:])
                    ax[i,j].plot(x,Ra_daily_org.view(-1).to('cpu').numpy(),label = 'Ra',color ='red')
                    #ax[i,j].set_title("Daily response")
                    if j == 0:
                        ax[i,j].set_ylabel("Y1_mean")
                    ax[i,j].set_xlabel(feature_name)
                    ax[i,j].legend()
                    ax1[i,j].plot(x,Rh_daily_org.view(-1).to('cpu').numpy(),label = 'Rh',color ='red')
                    #ax[i,j].set_title("Daily response")
                    if j == 0:
                        ax1[i,j].set_ylabel("Y1_mean")
                    ax1[i,j].set_xlabel(feature_name)
                    ax1[i,j].legend()

        plt.show()


if __name__ == "__main__":

    root_dir = 'E:/PyKGML/deposit_code_v2/'
    data_path = root_dir +  'processed_data/'
    output_path = root_dir + 'test_results/'

    #input_data = 'recotest_data_scaled_v4_100sample.sav'
    #sample_index_file = "flux_split_year_v1.sav"

    pretrained_model = "recotest_v11_exp4.sav_step4"
    output_model = "recotest_v11_exp4_sample.sav_step5"
    synthetic_data = "sys_data10.sav"

    dataset = Step5_DataSet(data_path, output_path)
    dataset.load_scaler_data('recotest_data_scaled_v4_scalers.sav')
    dataset.load_fluxtower_inputs_data('fluxtower_inputs_noscale_v2.sav')
    dataset.load_fluxtower_observe_data('fluxtower_observe_noscale_v2.sav')

    dataset.prepare_data('flux_split_year_v1.sav')

    model = Kgml_Model(output_path, output_path, pretrained_model, output_model, synthetic_data, dataset= dataset)
    model.load_RecoGRU_multitask_v11_3()

    model.load_synthetic_data()

    model.get_Ra_Rh_daily()

    #model.train_step5()

    model.vis_Ra_Rh()
