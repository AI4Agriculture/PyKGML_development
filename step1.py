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
####import kgml_lib
import kgml_lib

########################################################################################
###define functions from kgml_lib 
########################################################################################
Z_norm = kgml_lib.Z_norm 
Z_norm_reverse = kgml_lib.Z_norm_reverse
get_gpu_memory = kgml_lib.get_gpu_memory
my_loss = kgml_lib.my_loss
def stop_program():
    raise SystemExit("Program terminated due to using CPU.")

def train(data_path:str, output_path:str, input_data:str, sample_index_file:str, output_model:str):
    
    ########################################################################################
    ###prepare input and output data. Data will be 1) loaded, 2) normalized, and 3) put into (X, Y). 
    ###The data structure and input name are printed.
    ########################################################################################
    start=2001
    end=2018
    Tx=365 #timesteps
    tyear=end-start+1
    out1_names=['Ra','Rh','NEE']
    n_out1=len(out1_names)
    out2_names=['Yield']
    n_out2=len(out2_names)
    #time series data name
    fts_names = ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP','Ra','Rh','GrainC']
    #SP data name
    fsp_names = ['TBKDS','TSAND','TSILT','TFC','TWP','TKSat','TSOC','TPH','TCEC']
    f_names=['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP']+['Year']+fsp_names
    n_f=len(f_names)

    #load data
    #load in 100 samples from full sites 
    path_load = data_path+ input_data #'recotest_data_scaled_v4_100sample.sav'
    data0=torch.load(path_load)
    FIPS_ref = data0['FIPS_ref']
    bsz0 = len(FIPS_ref)
    #initial input and output
    X=torch.zeros([Tx*tyear,bsz0,n_f])
    X_scaler = np.zeros([n_f,2])
    Y1=torch.zeros([Tx*tyear,bsz0,n_out1])
    Y1_scaler = np.zeros([n_out1,2])
    Y2=torch.zeros([tyear,bsz0,n_out2])
    Y2_scaler = np.zeros([n_out2,2])

    #load in X variables
    X[:,:,0:9] = data0['X'][:,:,0:9]
    X_scaler[0:9,:] = data0['X_scaler'][0:9,:]
    #yead
    for y in range(tyear):
        X[y*Tx:(y+1)*Tx,:,9] = y+start
    range0=0
    range1=1
    X[:,:,9], X_scaler[9,0], X_scaler[9,1] = Z_norm(X[:,:,9])
    for i in range(len(fsp_names)):
        X[:,:,10+i] = data0['Xsp'][:,i].view(1,bsz0).repeat(Tx*tyear,1)
        X_scaler[10+i,:] = data0['Xsp_scaler'][i,:]
        
        
    #load in Y1
    Y1_scaler[0:2,:] = data0['X_scaler'][9:11,:]
    for i in range(2):
        Y1[:,:,i]= Z_norm_reverse(data0['X'][:,:,9+i],Y1_scaler[i,:],1.0)
    GPP = Z_norm_reverse(X[:,:,8],X_scaler[8,:],1.0)
    #GPP -Ra-Rh, Ra, Ra are negative,GPP +Ra+Rh+NEE =0
    Y1[:,:,2] = -(GPP+Y1[:,:,0]+Y1[:,:,1])
    for i in range(3):
        Y1[:,:,i], Y1_scaler[i,0], Y1_scaler[i,1] = Z_norm(Y1[:,:,i])

    #load in Y2
    Y2_scaler[:,:] = data0['X_scaler'][11,:]
    for y in range(tyear):
        Y2[y,:,0] = Z_norm_reverse(data0['X'][(y+1)*Tx-2,:,11],Y2_scaler[0,:],1.0)
    Y2[:,:,0],Y2_scaler[0,0],Y2_scaler[0,1] = Z_norm(Y2[:,:,0])

    #calculate the fraction of Res to GPP
    GPP_annual_all = torch.zeros([tyear,bsz0])
    Ra_annual_all = torch.zeros([tyear,bsz0])
    for y in range(tyear):
        GPP_annual_all[y,:] = torch.sum(Z_norm_reverse(X[y*Tx:(y+1)*Tx,:,8],X_scaler[8,:],1.0),dim=0)
        Ra_annual_all[y,:] = torch.sum(Z_norm_reverse(Y1[y*Tx:(y+1)*Tx,:,0],Y1_scaler[0,:],1.0),dim=0)
    Res_annual_all = GPP_annual_all + Ra_annual_all - Z_norm_reverse(Y2[:,:,0],Y2_scaler[0,:],1.0)
    GPP_Res_f = torch.mean(GPP_annual_all,dim=0)/torch.mean(Res_annual_all,dim=0)
    GPP_Res_fmean = GPP_Res_f.mean()
    Res_scaler = np.zeros([1,2])
    #feature scaling of Res
    Res__, Res_scaler[0,0], Res_scaler[0,1] = Z_norm(Res_annual_all)
    #print(GPP_Res_fmean,Res_scaler)

    print(X.size(),Y1.size(),Y2.size())
    print(f_names)

    #load gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        ###check GPU memory
        get_gpu_memory()
    else:
        device = torch.device("cpu")
    print(device)

    ########################################################################################
    ###prepare the validation samples by random sampling two years from every site rest will
    ### be used for training. The device we have and validation sample index are printed
    ########################################################################################
    batch_first = True
    #change to batch first
    if batch_first == True:
        X_train = torch.from_numpy(np.einsum('ijk->jik', X.numpy()))
        Y1_train = torch.from_numpy(np.einsum('ijk->jik', Y1.numpy()))
        Y2_train = torch.from_numpy(np.einsum('ijk->jik', Y2.numpy()))

    #sample data, all data
    train_n = X_train.size(0)
    #create mask for each data point. Random choose 2 year maske 0
    sample_index = np.random.randint(18, size=(train_n,2))
    print(sample_index)
    if not os.path.exists (data_path + sample_index_file): #'traindataset_split_year_v1.sav'
        torch.save({'sample_index':sample_index,
                },data_path + sample_index_file) #'traindataset_split_year_v1.sav'
    else:
        tmp = torch.load(data_path + sample_index_file) #'traindataset_split_year_v1.sav'
        sample_index = tmp['sample_index']

    #create mask for splitting training/validation
    Y1_mask_train=torch.zeros(Y1_train.size())+1.0
    Y2_mask_train=torch.zeros(Y2_train.size())+1.0
    Y1_mask_val=torch.zeros(Y1_train.size())
    Y2_mask_val=torch.zeros(Y2_train.size())
    for i in range(train_n):
        for y in range(2):
            Y1_mask_train[i,sample_index[i,y]*Tx:(sample_index[i,y]+1)*Tx,:] = 0.0
            Y1_mask_val[i,sample_index[i,y]*Tx:(sample_index[i,y]+1)*Tx,:] = 1.0
        Y2_mask_train[i,sample_index[i,:],:] = 0.0
        Y2_mask_val[i,sample_index[i,:],:] = 1.0
    Y1_maskb_val = Y1_mask_val.ge(0.5)
    Y2_maskb_val = Y2_mask_val.ge(0.5)

    ########################################################################################
    ###Create the KGML model structure. Model modules are going to be presented. 
    ###Parameters for each module will be presented.
    ########################################################################################
    seq_len=365
    model1= kgml_lib.RecoGRU_multitask_v11_3(n_f,0.2,seq_len,X_scaler[8,:],\
                                            Y1_scaler[0,:],Y2_scaler[0,:],Res_scaler[0,:])
    model1.to(device)
    print(model1)
    params = list(model1.parameters())
    print(len(params))
    print(params[5].size())  # conv1's .weight
    print("Model's state_dict:")
    for param_tensor in model1.state_dict():
        print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    compute_r2=kgml_lib.R2Loss()
    loss_weights = [1.0,1.0,1.0]
    lamda = [1.0,1.0]
    tol_mb = 0.01
    #self paced with a start threshold  and increase every epoch
    SPL_1= kgml_lib.SPLoss_YieldRa(1, 1.02) ####if use all data, 0.1,1.3

    ########################################################################################
    ###Model is going to be trained here with self-paced loss! However, we are not being able
    ### to really run since no GPU. 
    ### Yield and Ra modules are going to be trained together, while other parts are frozen.
    ###The weights going to be trained will be printed. 
    ###The model will be terminated with an error printed.
    ########################################################################################

    ##############################1) train yield and Ra
    ##freeze Rh and NEE
    for name, param in model1.named_parameters():
        param.requires_grad = True
        if param.requires_grad and ('gru_Rh' in name or 'densor_Rh' in name or 'gru_NEE' in name \
                                    or 'densor_NEE' in name):
            param.requires_grad = False
        if param.requires_grad:
            print(name)
    #initials
    loss_val_best = 500000
    R2_best=0.5
    best_epoch = 1000

    #train the model and pring loss/ yearly training
    starttime=time.time()
    lr_adam=0.001 #orginal 0.0001
    optimizer = optim.Adam(model1.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #bsz = 500
    bsz = 50 ###sample
    totsq=Tx*tyear
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
    maxepoch=1000

    ####################################
    if not torch.cuda.is_available():
        print("Cuda not avaliable")
        stop_program()
    ###Only run following codes when with a GPU

    # Output model 
    model_version= output_model #'recotest_v11_exp4_sample.sav1'
    path_save = output_path+model_version

    #################################training
    model1.train()
    for epoch in range(maxepoch):
        train_loss=0.0
        train_loss1=0.0
        train_loss2=0.0
        val_loss=0.0
        #shuffled the training data
        shuffled_b=torch.randperm(X_train.size()[0]) 
        X_train_new=X_train[shuffled_b,:,:] 
        Y1_train_new=Y1_train[shuffled_b,:,:]
        Y2_train_new=Y2_train[shuffled_b,:,:]
        Y1_mask_train_new = Y1_mask_train[shuffled_b,:,:]
        Y2_mask_train_new = Y2_mask_train[shuffled_b,:,:]
        Y1_maskb_train_new = Y1_mask_train_new.ge(0.5)
        Y2_maskb_train_new = Y2_mask_train_new.ge(0.5)
        Y1_pred_all=torch.zeros(Y1_train_new.size(),device=device)
        Y2_pred_all=torch.zeros(Y2_train_new.size(),device=device)
        model1.zero_grad()
        val_flag = False
        for bb in range(int(train_n/bsz)):
            if bb != int(train_n/bsz)-1:
                sbb = bb*bsz
                ebb = (bb+1)*bsz
            else:
                sbb = bb*bsz
                ebb = train_n
            #model initials
            with torch.no_grad():
                hidden = model1.init_hidden(ebb-sbb)
                ##spinup 2 years for turnover
                for it in range(2):
                    __,___,hidden = model1(X_train_new[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
            for it in range(maxit):
                #print(sbb,ebb,X_train_new[slw05*it:slw05*it+slw,sbb:ebb,:].size(),hidden.size(),X_train_new.size())
                Y1_pred,Y2_pred,hidden = model1(X_train_new[sbb:ebb,slw05*it:slw05*it+slw,:].to(device),hidden)
                loss,loss1,loss2 = SPL_1(Y1_pred[:,:,0]*Y1_mask_train_new[sbb:ebb,slw05*it:slw05*it+slw,0].to(device), \
                                        Y1_train_new[sbb:ebb,slw05*it:slw05*it+slw,0].to(device)*\
                                        Y1_mask_train_new[sbb:ebb,slw05*it:slw05*it+slw,0].to(device), \
                                Y2_pred.view(ebb-sbb,1)*Y2_mask_train_new[sbb:ebb,it,0].view(ebb-sbb,1).to(device),\
                                        Y2_train_new[sbb:ebb,it,:].view(ebb-sbb,1).to(device)*\
                                        Y2_mask_train_new[sbb:ebb,it,0].view(ebb-sbb,1).to(device),val_flag)
                for zz in range(len(hidden)):
                    hidden[zz].detach_()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_loss=train_loss+loss.item()
                    train_loss1=train_loss1+loss1.item()
                    train_loss2=train_loss2+loss2.item()
                    Y1_pred_all[sbb:ebb,slw05*it:slw05*it+slw,0]=Y1_pred[:,:,0]
                    Y2_pred_all[sbb:ebb,it,0]=Y2_pred[:,0,0]
        scheduler.step()
        #update self-paced loss threshold
        if epoch < 500:
            SPL_1.increase_threshold()
        #################################validation
        model1.eval()
        val_flag = True
        with torch.no_grad():
            train_loss=[train_loss/(maxit*train_n/bsz),train_loss1/(maxit*train_n/bsz),\
                        train_loss2/(maxit*train_n/bsz)]
            train_losses.append(train_loss)
            #R2 for all
            train_R2 = []
            for varn in range(1): 
                Y1_pred_all_masked=torch.masked_select(Y1_pred_all[:,:,varn], Y1_maskb_train_new[:,:,varn].to(device))
                Y1_train_new_masked=torch.masked_select(Y1_train_new[:,:,varn], Y1_maskb_train_new[:,:,varn])
                train_R2.append(compute_r2(Y1_pred_all_masked.contiguous().view(-1),\
                                        Y1_train_new_masked.contiguous().view(-1).to(device)).item())
            for varn in range(n_out2):
                Y2_pred_all_masked=torch.masked_select(Y2_pred_all[:,:,varn], Y2_maskb_train_new[:,:,varn].to(device))
                Y2_train_new_masked=torch.masked_select(Y2_train_new[:,:,varn], Y2_maskb_train_new[:,:,varn])
                train_R2.append(compute_r2(Y2_pred_all_masked.contiguous().view(-1),\
                                        Y2_train_new_masked.contiguous().view(-1).to(device)).item())
            Y1_val_pred=torch.zeros(Y1_train.size(),device=device)
            Y2_val_pred=torch.zeros(Y2_train.size(),device=device)
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
                    Y1_val_pred[sbb:ebb,slw05_val*it:slw05_val*it+slw_val,0] = Y1_val_pred_t[:,:,0]
                    Y2_val_pred[sbb:ebb,it,0] = Y2_val_pred_t[:,0,0]
            loss,loss1,loss2 = SPL_1(Y1_val_pred[:,:,0]*Y1_mask_val[:,:,0].to(device), \
                                    Y1_train[:,:,0].to(device)*Y1_mask_val[:,:,0].to(device), \
                                Y2_val_pred[:,:,0]*Y2_mask_val[:,:,0].to(device), \
                                    Y2_train[:,:,0].to(device)*Y2_mask_val[:,:,0].to(device), val_flag)
            val_loss=[loss.item(),loss1.item(),loss2.item()]
            val_losses.append(val_loss)
            #R2 for all
            val_R2= []
            for varn in range(1):
                Y1_val_pred_masked=torch.masked_select(Y1_val_pred[:,:,varn], Y1_maskb_val[:,:,varn].to(device))
                Y1_val_masked=torch.masked_select(Y1_train[:,:,varn], Y1_maskb_val[:,:,varn])    
                val_R2.append(compute_r2(Y1_val_pred_masked.contiguous().view(-1),\
                                        Y1_val_masked.contiguous().view(-1).to(device)).item())
            for varn in range(n_out2):
                Y2_val_pred_masked=torch.masked_select(Y2_val_pred[:,:,varn], Y2_maskb_val[:,:,varn].to(device))
                Y2_val_masked=torch.masked_select(Y2_train[:,:,varn], Y2_maskb_val[:,:,varn])             
                val_R2.append(compute_r2(Y2_val_pred_masked.contiguous().view(-1),\
                                        Y2_val_masked.contiguous().view(-1).to(device)).item())
            #if val_loss < loss_val_best and val_R2 > R2_best:
            if val_loss[1]+val_loss[2] < loss_val_best:
                loss_val_best=val_loss[1]+val_loss[2]
                R2_best = val_R2
                best_epoch = epoch
                f0=open(path_save,'w')
                f0.close()
                #os.remove(path_save)
                torch.save({'epoch': epoch,
                        'model_state_dict': model1.state_dict(),
                        'R2': train_R2,
                        'loss': train_loss,
                        'los_val': val_loss,
                        'R2_val': val_R2,
                        }, path_save)    
            print("finished training epoch", epoch+1)
            mtime=time.time()
            print("train_loss: ", train_loss, "train_R2", train_R2,"val_loss:",val_loss,"val_R2", val_R2,\
                "loss val best:",loss_val_best,"R2 val best:",R2_best, f"Spending time: {mtime - starttime}s")

            if np.mean(train_R2)> 0.999 or (epoch - best_epoch) > 100:
                break
        model1.train()
    endtime=time.time()
    path_fs = path_save+'fs1'
    torch.save({'train_losses': train_losses,
                'val_losses': val_losses,
                'model_state_dict_fs': model1.state_dict(),
                }, path_fs)
    
    print("Step 1 saved model is: ", path_fs)
    print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best)