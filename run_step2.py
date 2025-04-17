import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from io import open
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from scipy.stats import gaussian_kde
import scipy.stats as stats

import kgml_lib

from dataset import Step2_DataSet


# define functions from kgml_lib
Z_norm = kgml_lib.Z_norm 
Z_norm_reverse = kgml_lib.Z_norm_reverse
get_gpu_memory = kgml_lib.get_gpu_memory
my_loss = kgml_lib.my_loss
compute_r2=kgml_lib.R2Loss()

myloss_mb_flux_mask = kgml_lib.myloss_mb_flux_mask
check_Rh2SOC_response = kgml_lib.check_Rh2SOC_response

def stop_program():
    raise SystemExit("Program terminated due to using CPU.")

class Kgml_Model:
    def __init__(self, input_path:str, output_path:str, pretrained_model:str, output_model:str, synthetic_data:str, dataset:Step2_DataSet):
        self.input_path = input_path
        self.output_path = output_path
        self.pretrained_model = input_path + pretrained_model
        self.output_model = output_model
        self.path_save = output_path + output_model
        self.dataset = dataset

        self.synthetic_data = self.input_path + synthetic_data

        # self.loss_weights = [1.0,1.0,1.0]
        # self.lamda = [1.0,1.0]
        # self.tol_mb = 0.01

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

    def load_synthetic_data(self):
        if os.path.exists(self.synthetic_data):
            tmp = torch.load(self.synthetic_data, weights_only=False) #'sys_data2.sav'
            self.check_xset = tmp['sys_data2']
        else:
            raise FileNotFoundError(f"Synthetic_data {self.synthetic_data} not exist.")

    def freeze_yield(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            if param.requires_grad and ('attn' in name or 'gru_basic' in name or 'densor_yield' in name):
                param.requires_grad = False
            if param.requires_grad:
                print(name)

    def train_step2(self, LR=0.001,step_size=20, gamma=0.6, maxepoch=80):
        
        self.load_synthetic_data()
        self.freeze_yield()

        #initials
        loss_val_best = 500000
        R2_best=0.5
        best_epoch = 1500
        loss_weights = [1.0,1.0,1.0]
        lamda = [1.0,1.0]
        tol_mb = 0.01

        starttime=time.time()
        lr_adam= LR #0.001 #orginal 0.0001
        optimizer = optim.Adam(self.model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        #bsz = 500
        bsz =50 ###for sample test
        totsq=self.dataset.Tx * self.dataset.tyear # 365*18=6570
        #during training
        slw=365
        slw05=365
        maxit=int((totsq-slw)/slw05+1)
        #during validation
        slw_val=slw05 #use step size to predict
        slw05_val=slw05
        maxit_val=int((totsq-slw_val)/slw05_val+1)
        train_losses = []
        val_losses = []
        maxepoch=maxepoch

        device = torch.device("cuda")

        model1 = self.model

        X_train = self.dataset.X_train
        Y1_train =self.dataset.Y1_train
        
        Y1_mask_train = self.dataset.Y1_mask_train
        Y1_mask_val = self.dataset.Y1_mask_val
        Y1_maskb_val = self.dataset.Y1_maskb_val

        train_n = X_train.size()[0]

        GPP_index = self.dataset.fts_names_1.index('GPP') #8
        GPP_scaler = self.dataset.X_scaler[GPP_index,:]

        Y1_scaler = self.dataset.Y1_scaler

        #model1.train()
        for epoch in range(maxepoch):
            train_loss=0.0
            train_loss1=0.0
            train_loss2=0.0
            train_loss3=0.0
            val_loss=0.0
            #shuffled the training data, shuffle site ID
            shuffled_b=torch.randperm(X_train.size()[0]) 
            X_train_new=X_train[shuffled_b,:,:] 
            Y1_train_new=Y1_train[shuffled_b,:,:]
            Y1_mask_train_new = Y1_mask_train[shuffled_b,:,:]
            Y1_maskb_train_new = Y1_mask_train_new.ge(0.5)
            Y1_pred_all=torch.zeros(Y1_train_new.size(),device=device)

            model1.train()
            model1.zero_grad()
            for bb in range(int(train_n/bsz)): # 100/50 = 2
                if bb != int(train_n/bsz)-1:
                    sbb = bb*bsz
                    ebb = (bb+1)*bsz
                else:
                    sbb = bb*bsz
                    ebb = train_n
                ####train the loss3 once before into the maxit to save time
                if epoch > 0:
                    loss3 = check_Rh2SOC_response(model1, self.check_xset,Y1_scaler,device)
                    optimizer.zero_grad()
                    loss3.backward()
                    optimizer.step()  
                    with torch.no_grad():
                        train_loss3=train_loss3+loss3.item()
                #model initials
                with torch.no_grad():
                    hidden = model1.init_hidden(ebb-sbb)
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(X_train_new[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
                
                for it in range(maxit):
                    #print(sbb,ebb,X_train_new[slw05*it:slw05*it+slw,sbb:ebb,:].size(),hidden.size(),X_train_new.size())
                    Y1_pred,Y2_pred,hidden = model1(X_train_new[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
                    loss,loss1,loss2 = myloss_mb_flux_mask(Y1_pred,Y1_train_new[sbb:ebb,slw05*it:slw05*it+slw,:].to(device), 
                                    X_train_new[sbb:ebb,slw05*it:slw05*it+slw,GPP_index].to(device), 
                                    GPP_scaler, Y1_scaler,
                                    Y1_mask_train_new[sbb:ebb,slw05*it:slw05*it+slw,0].to(device), loss_weights, lamda,tol_mb)
                    for zz in range(len(hidden)):
                        hidden[zz].detach_()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        train_loss=train_loss+loss.item()
                        train_loss1=train_loss1+loss1.item()
                        train_loss2=train_loss2+loss2.item()
                        Y1_pred_all[sbb:ebb,slw05*it:slw05*it+slw,:]=Y1_pred[:,:,:]
            scheduler.step()

            #validation
            model1.eval()
            with torch.no_grad():
                train_loss=[train_loss/(maxit*train_n/bsz),train_loss1/(maxit*train_n/bsz),\
                            train_loss2/(maxit*train_n/bsz),train_loss3/(train_n/bsz)]
                train_losses.append(train_loss)
                #R2 for all
                train_R2 = []
                for varn in range(self.dataset.n_out1): 
                    Y1_pred_all_masked=torch.masked_select(Y1_pred_all[:,:,varn], Y1_maskb_train_new[:,:,varn].to(device))
                    Y1_train_new_masked=torch.masked_select(Y1_train_new[:,:,varn], Y1_maskb_train_new[:,:,varn])
                    train_R2.append(compute_r2(Y1_pred_all_masked.contiguous().view(-1),\
                                            Y1_train_new_masked.contiguous().view(-1).to(device)).item())
                
                Y1_val_pred=torch.zeros(Y1_train.size(),device=device)
                for bb in range(int(train_n/bsz)):
                    if bb != int(train_n/bsz)-1:
                        sbb = bb*bsz
                        ebb = (bb+1)*bsz
                    else:
                        sbb = bb*bsz
                        ebb = train_n
                    hidden = model1.init_hidden(ebb-sbb)
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(X_train[sbb:ebb,slw05_val*it:slw05_val*it+slw_val,:].to(device),hidden)
                    for it in range(maxit_val):
                        #change intitial for year round simulation
                        Y1_val_pred_t,Y2_val_pred_t, hidden = model1(X_train[sbb:ebb,slw05_val*it:slw05_val*it+slw_val,:].to(device),\
                                                                    hidden)
                        Y1_val_pred[sbb:ebb,slw05_val*it:slw05_val*it+slw_val,:] = Y1_val_pred_t
                
                loss,loss1,loss2 = myloss_mb_flux_mask(Y1_val_pred, Y1_train.to(device),\
                                    X_train[:,:,GPP_index].to(device), GPP_scaler, \
                                    Y1_scaler,Y1_mask_val[:,:,0].to(device),loss_weights, lamda,tol_mb)
                loss3 = check_Rh2SOC_response(model1, self.check_xset,Y1_scaler,device)
                loss = loss + loss3
                val_loss=[loss.item(),loss1.item(),loss2.item(),loss3.item()]
                val_losses.append(val_loss)
                #R2 for all
                val_R2= []
                for varn in range(self.dataset.n_out1):
                    Y1_val_pred_masked=torch.masked_select(Y1_val_pred[:,:,varn], Y1_maskb_val[:,:,varn].to(device))
                    Y1_val_masked=torch.masked_select(Y1_train[:,:,varn], Y1_maskb_val[:,:,varn])    
                    val_R2.append(compute_r2(Y1_val_pred_masked.contiguous().view(-1),\
                                            Y1_val_masked.contiguous().view(-1).to(device)).item())
                
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
                print("train_loss: ", train_loss, "train_R2", train_R2,"val_loss:",val_loss,"val_R2", val_R2,\
                    "loss val best:",loss_val_best,"R2 val best:",R2_best, f"Spending time: {mtime - starttime}s")

                if np.mean(train_R2)> 0.999 or (epoch - best_epoch) > 10:
                    print(f"Exit condition: mean R2 {np.mean(train_R2)}, epoch {epoch}, best_epoch {best_epoch}")
                    break
            model1.train()

        endtime=time.time()
        path_fs = self.path_save +'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model1.state_dict(),
                    }, path_fs)
        print("Step 2 saved model is: ", path_fs)
        print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best)
        print(f"total Training time: {endtime - starttime}s")

    def check_results(self,device,model_trained,total_b,bsz,check_xset,check_y1set,check_y2set,slw05,slw,maxit,starttime):
        Y1_pred=torch.zeros(check_y1set.size(),device=device)
        Y2_pred=torch.zeros(check_y2set.size(),device=device)
        #print(total_b,bsz)
        for bb in range(int(total_b/bsz)):
            if bb != int(total_b/bsz)-1:
                sbb = bb*bsz
                ebb = (bb+1)*bsz
            else:
                sbb = bb*bsz
                ebb = total_b
                    #model initials
            with torch.no_grad():
                hidden = model_trained.init_hidden(ebb-sbb)
                ##spinup 2 years for turnover
                for it in range(2):
                    __,___,hidden = model_trained(check_xset[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
            for it in range(maxit):
                    #change intitial for year round simulation
                Y1_pred_t,Y2_pred_t, hidden = model_trained(check_xset[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
                Y1_pred[sbb:ebb,slw05*it:slw05*it+slw,:] = Y1_pred_t[:,:,:]
                Y2_pred[sbb:ebb,it,:] = Y2_pred_t[:,:,0]
                mtime=time.time()
                print("finished,",bb, it, f"Spending time: {mtime - starttime}s")
        return Y1_pred,Y2_pred

    def test_step2(self, model_version, device):
        starttime=time.time()
        slw=365
        slw05=365
        data = self.dataset
        totsq=data.Tx * data.tyear
        maxit=int((totsq-slw)/slw05+1)
        compute_r2=kgml_lib.R2Loss()
        n_a=64 #hidden state number
        n_l=2 #layer of lstm
        dropout=0.2
        seq_len = 365
        ####read in a pretrained model with full sample
        #model_version='recotest_v11_exp4.sav_step2'  #####!!!!!!!!!!!!!!!!!!!! change this before training
        path_save = self.output_path + model_version
        model_version_t=model_version
        with torch.no_grad():
            checkpoint=torch.load(path_save, map_location=device,weights_only=False)
            model_trained=kgml_lib.RecoGRU_multitask_v11_3(data.n_f,dropout,seq_len,data.X_scaler[8,:],data.Y1_scaler[0,:],data.Y2_scaler[0,:],data.Res_scaler[0,:])
            model_trained.load_state_dict(checkpoint['model_state_dict'])
            model_trained.to(device) #too large for GPU, kif not enough, change to cpu
            model_trained.eval()
            epoch = checkpoint['epoch']
            print(epoch)
        
            bsz = 1  # this is the batch size for training
            #total_b = train_n ##test when using GPU
            total_b = 2 ##test when using CPU
            
            #predict with the model
            data.Y1_train_pred, data.Y2_train_pred=  self.check_results(device,model_trained,total_b,bsz,
                    data.X_train[0:total_b,:,:],data.Y1_train[0:total_b,:,:],data.Y2_train[0:total_b,:,:],
                    slw05,slw,maxit,starttime) 
            

        endtime=time.time()
        print(f"Total spending time: {endtime - starttime}s")

    def vis_loss(self, model_version):
        ########################################################################################
        ###visualize your model training losses vs validation losses
        ########################################################################################
        plt.rcParams.update({'font.size': 20})
        plt.rcParams['xtick.labelsize']=20
        plt.rcParams['ytick.labelsize']=20
        #fig, ax = plt.subplots(4,1,figsize=(7*1, 5*4))
        fig, ax = plt.subplots(2,2,figsize=(5*4,12*1))
        lw = 3.0
        plot_names = ['All','FLUX MSE','Mass_balance','Response_Rh2SOC']
        path_save = self.output_path+model_version+ 'fs'  #'fs_2'
        checkpoint=torch.load(path_save, map_location=torch.device('cpu'),weights_only=False)
        train_losses2=checkpoint['train_losses']
        val_losses2=checkpoint['val_losses']
        train_plots2 = np.zeros([len(train_losses2),len(train_losses2[0])])
        val_plots2 = np.zeros([len(train_losses2),len(train_losses2[0])])
        for i in range(len(train_losses2)):
            train_plots2[i,:] = np.array(train_losses2[i])
            val_plots2[i,:] = np.array(val_losses2[i])
        for i in range(2):
            for j in range(2):
                ax[i,j].plot(train_plots2[:,i], label="Train loss",lw=lw)
                ax[i,j].plot(val_plots2[:,i], label="Val loss",lw=lw)

                #ax[j,i].set_ylim([0,zoomin[j]])
                ax[i,j].set_ylabel(plot_names[i*2+j], fontweight='bold')
                ax[i,j].set_xlabel("Epoch")
                #if i==1:
                #    ax[i,j].set_xlabel("Epoch")
                ax[i,j].legend()
        plt.show()


    def vis_yield_prediction(self):
        ########################################################################################
        ###visualize yield prediction vs synthetic ground truth. 
        ########################################################################################
        data = self.dataset
        total_b = 2 # same as in test_step2()

        Y2_maskb_train = data.Y2_mask_train[0:total_b,:,:].ge(0.5)
        Y2_maskb_val = data.Y2_mask_val[0:total_b,:,:].ge(0.5)

        plt.rcParams.update({'font.size': 13})
        plt.rcParams['xtick.labelsize']=13
        plt.rcParams['ytick.labelsize']=13
        ncols=2
        nrows=3
        units_convert=[1.0]
        fig, ax = plt.subplots(nrows,ncols,figsize=(6*ncols, 5*nrows))   
        varn = 0
        ct_index = [0,1,5]
        plot_names = ['Yield','Corn Yield','Soybean Yield']
        test_names = ['Train','Test']
        for col in range(2):
            
            # for all data
            #read in crop_type
            crop_type = torch.from_numpy(np.zeros([data.X_train[0:total_b,:,:].size(0),18]).astype(int))
            for y in range(18):
                crop_type[:,y] = Z_norm_reverse(data.X_train[0:total_b,y*365+1,7],data.X_scaler[7,:],1.0).int()
            print(crop_type.shape)
            Ysim=Z_norm_reverse(data.Y2_train_pred[0:total_b,:,varn],data.Y2_scaler[varn,:],units_convert[varn]).to("cpu")
            Yobs=Z_norm_reverse(data.Y2_train[0:total_b,:,varn],data.Y2_scaler[varn,:],units_convert[varn])
            if col == 0:
                Y_maskb = Y2_maskb_train[:,:,varn]
            else:
                Y_maskb = Y2_maskb_val[:,:,varn]
            Yobs_masked = torch.masked_select(Yobs, Y_maskb)
            Ysim_masked = torch.masked_select(Ysim, Y_maskb)
            crop_type_masked = torch.masked_select(crop_type, Y_maskb)
            for i in range(3):
                if i == 0:
                    ax[i,col].scatter( Yobs_masked.contiguous().view(-1).numpy(),Ysim_masked.contiguous().view(-1).numpy(),\
                                    s=10,color='black',alpha=0.5)
                    R2 = compute_r2(Ysim_masked, Yobs_masked).numpy()
                    RMSE =  np.sqrt(my_loss(Ysim_masked, Yobs_masked).numpy())
                    Bias = torch.mean(Ysim_masked-Yobs_masked).numpy()
                    m, b, r_value, p_value, std_err = stats.linregress(Ysim_masked.contiguous().view(-1).numpy(), \
                                                                    Yobs_masked.contiguous().view(-1).numpy()) #r,p,std
                else:
                    ax[i,col].scatter( Yobs_masked[crop_type_masked == ct_index[i]].contiguous().view(-1).numpy(),\
                                    Ysim_masked[crop_type_masked == ct_index[i]].contiguous().view(-1).numpy(),\
                                    s=10,color='black',alpha=0.5)
                    R2 = compute_r2(Ysim_masked[crop_type_masked == ct_index[i]], Yobs_masked[crop_type_masked == ct_index[i]]).numpy()
                    RMSE =  np.sqrt(my_loss(Ysim_masked[crop_type_masked == ct_index[i]], \
                                            Yobs_masked[crop_type_masked == ct_index[i]]).numpy())
                    Bias = torch.mean(Ysim_masked[crop_type_masked == ct_index[i]]-\
                                    Yobs_masked[crop_type_masked == ct_index[i]]).numpy() 
                    m, b, r_value, p_value, std_err = stats.linregress(Ysim_masked[crop_type_masked == ct_index[i]].\
                                                                    contiguous().view(-1).numpy(), \
                                    Yobs_masked[crop_type_masked == ct_index[i]].contiguous().view(-1).numpy()) #r,p,std

                    #lim_min=min(min(Ysim),min(Yobs))

                    #lim_max=max(max(Ysim),max(Yobs))
                lim_min=0
                lim_max=800
                ax[i,col].set_xlim([lim_min,lim_max])
                ax[i,col].set_ylim([lim_min,lim_max])
                ax[i,col].text(50,600,'R$^2$=%0.3f\nRMSE=%0.3f\nbias=%0.3f' 
                    % (R2,RMSE,Bias),fontsize = 12)
                if i == nrows-1:
                    ax[i,col].set_xlabel('Ecosys simulated yield',fontsize = 15,weight='bold')
                if col == 0:
                    ax[i,col].set_ylabel('GRU_yield',fontsize = 15,weight='bold')
                ax[i,col].set_title(plot_names[i]+' ('+test_names[col]+')',fontsize = 15,weight='bold')
                ax[i,col].plot(Yobs_masked, m*Yobs_masked + b,color='steelblue',lw=1.0)
                ax[i,col].plot([lim_min,lim_max], [lim_min,lim_max],color='red',linestyle='--')
                
        plt.show()

    def vis_flux_prediction(self):
        ########################################################################################
        ###visualize flux prediction vs synthetic ground truth. Note that we only trained Ra
        ########################################################################################
        data = self.dataset
        total_b = 2

        Y1_maskb_val = data.Y1_mask_val[0:total_b,:,:].ge(0.5)
        Y2_maskb_val = data.Y2_mask_val[0:total_b,:,:].ge(0.5)

        plt.rcParams.update({'font.size': 13})
        plt.rcParams['xtick.labelsize']=13
        plt.rcParams['ytick.labelsize']=13
        ncols=2
        nrows=3
        units_convert=[1.0]
        fig, ax = plt.subplots(nrows,ncols,figsize=(6*ncols, 5*nrows))   
        varn = 0
        ct_index = [0,1,5]
        plot_names = ['Ra','Rh','NEE']
        test_names = ['Daily', 'Annual']
        for col in range(ncols):
            check_set = data.Y1_train[0:total_b,:,:]
            check_set_pred = data.Y1_train_pred
            if col == 0:
                Y_maskb = Y1_maskb_val[:,:,0]
            else: 
                Y_maskb = Y2_maskb_val[:,:,0]
            for i in range(3):
                
                if col == 0:
                    Ysim=Z_norm_reverse(check_set_pred[:,:,i],data.Y1_scaler[i,:],1.0).to("cpu")
                    Yobs=Z_norm_reverse(check_set[:,:,i],data.Y1_scaler[i,:],1.0)
                    Ysim = torch.masked_select(Ysim, Y_maskb)
                    Yobs = torch.masked_select(Yobs, Y_maskb)
                    ax[i,col].scatter(Yobs.contiguous().view(-1).numpy(),Ysim.contiguous().view(-1).numpy(), \
                                    s=10,color='black',alpha=0.5)
                    
                else:
                    Ysim=torch.zeros([check_set.size(0),18])
                    Yobs=torch.zeros([check_set.size(0),18])
                    for y in range(data.tyear):
                        Ysim[:,y]=torch.sum(Z_norm_reverse(check_set_pred[:,y*data.Tx:(y+1)*data.Tx,i],data.Y1_scaler[i,:],1.0).to("cpu"),dim=1)
                        Yobs[:,y]=torch.sum(Z_norm_reverse(check_set[:,y*data.Tx:(y+1)*data.Tx,i],data.Y1_scaler[i,:],1.0),dim=1)
                    Ysim = torch.masked_select(Ysim, Y_maskb)
                    Yobs = torch.masked_select(Yobs, Y_maskb)
                    ax[i,col].scatter(Yobs.contiguous().view(-1).numpy(),Ysim.contiguous().view(-1).numpy(), \
                                    s=10,color='black',alpha=0.5)
                print(col,i,Ysim.size(),Yobs.size())
                
                
                R2 = compute_r2(Ysim, Yobs).numpy()
                RMSE =  np.sqrt(my_loss(Ysim, Yobs).numpy())
                Bias = torch.mean(Ysim-Yobs).numpy()
                m, b, r_value, p_value, std_err = stats.linregress(Ysim.contiguous().view(-1).numpy(), \
                                                                    Yobs.contiguous().view(-1).numpy()) #r,p,std
                
                lim_min=min(torch.min(Ysim).numpy(),torch.min(Yobs).numpy())
                lim_max=max(torch.max(Ysim).numpy(),torch.max(Yobs).numpy())
                ax[i,col].set_xlim([lim_min,lim_max])
                ax[i,col].set_ylim([lim_min,lim_max])
                ax[i,col].text(lim_min+0.1*(lim_max-lim_min),lim_max-0.1*(lim_max-lim_min),'R$^2$=%0.3f\nRMSE=%0.3f\nbias=%0.3f' 
                    % (R2,RMSE,Bias),fontsize = 12,ha='left', va='top')
                if i == nrows-1:
                    ax[i,col].set_xlabel('Ecosys simulated',fontsize = 15,weight='bold')
                if col == 0:
                    ax[i,col].set_ylabel('KGML predicted ' + '(g C $m^{-2}$ $day^{-1}$)',fontsize = 15,weight='bold')
                ax[i,col].set_title(plot_names[i]+' ('+test_names[col]+')',fontsize = 15,weight='bold')
                ax[i,col].plot(Yobs, m*Yobs + b,color='steelblue',lw=1.0)
                ax[i,col].plot([lim_min,lim_max], [lim_min,lim_max],color='red',linestyle='--')
                
        plt.show()


if __name__ == "__main__":

    root_dir = 'E:/PyKGML/deposit_code_v2/'
    data_path = root_dir +  'processed_data/'
    output_path = root_dir + 'test_results/'

    input_data = 'recotest_data_scaled_v4_100sample.sav'
    sample_index_file = "traindataset_split_year_v1.sav"

    pretrained_model = "recotest_v11_exp4.sav_step1"
    output_model = "recotest_v11_exp4_sample.sav_step2"
    synthetic_data = "sys_data2.sav"

    dataset = Step2_DataSet(data_path, input_data, output_path, sample_index_file)
    dataset.load_step2_data()

    dataset.prepare_step2_data()

    #dataset.process_step2_X()
    #dataset.process_step2_Y()

    dataset.train_test_split(sample_index_file = sample_index_file)

    model = Kgml_Model(output_path, output_path, pretrained_model, output_model, synthetic_data, dataset= dataset)
    model.load_RecoGRU_multitask_v11_3()

    #model.train_step2()

    # For test

    device = torch.device("cuda")
    #device = torch.device('cpu')
    model.test_step2('recotest_v11_exp4.sav_step2',device)

    model.vis_loss('recotest_v11_exp4.sav_step2')

    model.vis_yield_prediction()

    model.vis_flux_prediction()