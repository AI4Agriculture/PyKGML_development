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
####import kgml_lib
import kgml_lib

def train(data_path:str, output_path:str, input_data:str, sample_index_file:str, 
          pretrained_model:str, output_model:str, synthetic_data:str, fluxtower_inputs:str, fluxtower_observe:str):
    
    #prepare previous scaler, model 
    Tx=365 #timesteps
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

    #load the scaler
    path_load = data_path + input_data #'recotest_data_scaled_v4_scalers.sav'
    data0=torch.load(path_load)
    X_scaler = data0['X_scaler']
    Y1_scaler = data0['Y1_scaler']
    Y2_scaler = data0['Y2_scaler']
    Res_scaler = data0['Res_scaler']
    GPP_Res_fmean = data0['GPP_Res_fmean']
    #load gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    lamda = [1.0,1.0]
    tol_mb = 0.01

    ##load synthetic data for GPP response calculations
    tmp = torch.load(output_path + synthetic_data) #'sys_data10.sav'
    check_xset = tmp['sys_data10']

    #####define functions from kgml_lib
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
    get_gpu_memory()

    #######Run for save Rh_daily_org
    model_version= pretrained_model #'recotest_v11_exp4.sav_step4'
    path_save = output_path+model_version
    checkpoint=torch.load(path_save)
    model1=kgml_lib.RecoGRU_multitask_v11_3(n_f,0.2,Tx,X_scaler[8,:],Y1_scaler[0,:],Y2_scaler[0,:],Res_scaler[0,:])
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.to(device) 
    model1.eval()

    rangess = [[-2, 2],[-3.5, 2.5],[-2.5, 5],[-1.5,3.5],[-1.5,7],[-2.0,5.5],[-0.3889,10.0],[-1,1],[-0.5,5],[-1.6384,1.6384],\
            [-10,2],[-0.75,4],[-3.25,1.25],[-4.5,5.0],[-2.5,3],[-0.75,7.75],[-0.5,5],[-3.25,4.25],[-1.75,3.5]]
    intervals = [0.2,0.3,0.5,0.25,0.5,0.5,0.5,2,0.25,1.6384*2/17,0.5,0.25,0.25,0.5,0.25,0.5,0.25,0.5,0.25]
    #########test different interval multiplier
    intv_f = 3.0
    ########set tolerance r for each variable
    Ra_r_mins = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    Rh_r_mins = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.99, 0.9, 0.9]

    Ra_daily_orgs =[]
    Rh_daily_orgs =[]
    for i in range(4):
        for j in range(5):
            if not (i==3 and j ==4):
                feature = i*5+j
                ranges = rangess[feature]
                #########test different interval multiplier
                if feature == 7:
                    interval = intervals[feature]
                else:
                    interval = intervals[feature]*intv_f
                total_n = int((ranges[1] - ranges[0])/interval)+1
                with torch.no_grad():
                    Ra_daily_org, Rh_daily_org = Get_Ra_Rh_org(model1,check_xset,feature, ranges, interval,Y1_scaler, device)
                    Ra_daily_orgs.append(Ra_daily_org)
                    Rh_daily_orgs.append(Rh_daily_org)

    ##########test performance
    starttime=time.time()
    #using all data and 2 year out for validation
    for cv_n in range(1):
        #prepare training and validation set
        
        #prepare data and mask
        #flux site level validation
        data_dir = data_path
        path_load = data_dir + fluxtower_inputs #'fluxtower_inputs_noscale_v2.sav'
        data0=torch.load(path_load)
        mode_input = data0['mode_input']
        path_load = data_dir + fluxtower_observe #'fluxtower_observe_noscale_v2.sav'
        data0=torch.load(path_load)
        org_Reco = data0['org_Reco']
        org_NEE = data0['org_NEE']
        Y_Reco_obs = []
        Y_NEE_obs = []
        X_sites = []
        Y_masks = []
        Y_masksb = []
        Y_masks_train = []
        Y_masks_val = []
        sample_indexes = []
        if os.path.exists (output_path + sample_index_file): #'flux_split_year_v1.sav'
            tmp = torch.load(output_path + sample_index_file) #'flux_split_year_v1.sav'
            sample_indexes = tmp['sample_indexes'] 
        #for i in range(1):
        for i in range(len(mode_input)):
            tyear1 = org_Reco[i].shape[0]
            totsq1=mode_input[i].size(1)
            #1) reshape the observed data to predicted scale
            Y_Reco_obs_t = np.zeros(totsq1)
            Y_NEE_obs_t = np.zeros(totsq1)
            Y_mask_t = np.zeros(totsq1) + 1.0
            for y in range(tyear1):
                Y_Reco_obs_t[y*365:(y+1)*365] = org_Reco[i][y,0:365]
                Y_NEE_obs_t[y*365:(y+1)*365]= org_NEE[i][y,0:365]
            #replace nan to 0 to avoid error
            nanindex = np.logical_or(np.isnan(Y_Reco_obs_t),np.isnan(Y_NEE_obs_t))
            Y_Reco_obs_t[nanindex] = 0.0 
            Y_NEE_obs_t[nanindex] = 0.0
            Y_mask_t[nanindex] = 0.0
            GPP_obs_t = Y_Reco_obs_t - Y_NEE_obs_t
            GPP_obs_t[nanindex] = mode_input[i][0,nanindex,8].numpy()
            nanindex = np.isnan(GPP_obs_t)
            GPP_obs_t[nanindex] = 0.0
            Y_mask_t[nanindex] = 0.0
            #corrected GPP
            mode_input[i][0,:,8] = torch.from_numpy(GPP_obs_t)
            Y_Reco_obs.append(torch.from_numpy(Y_Reco_obs_t).to(device))
            Y_NEE_obs.append(torch.from_numpy(Y_NEE_obs_t).to(device))
            Y_masks.append(torch.from_numpy(Y_mask_t).to(device))
            #2) scale th org model input
            X_site = torch.zeros(mode_input[i].size())
            for f in range(len(f_names)):
                X_site[:,:,f] = Z_norm_with_scaler(mode_input[i][:,:,f],X_scaler[f,:])
            X_sites.append(X_site.to(device))
            print(Y_masks[i].size(),X_sites[i].size())
            #######develop sample index and mask for train and validation
            sample_index = np.random.randint(tyear1, size=(2))
            if len(sample_indexes) > i:
                sample_index = sample_indexes[i]
            else:
                sample_indexes.append(sample_index)
            Y_mask_train=torch.zeros(Y_masks[i].size())+1.0
            Y_mask_val=torch.zeros(Y_masks[i].size())
            for yy in range(2):
                Y_mask_train[sample_index[yy]*365:(sample_index[yy]+1)*365] = 0.0
                Y_mask_val[sample_index[yy]*365:(sample_index[yy]+1)*365] = 1.0
            Y_masks_train.append(Y_mask_train.to(device)*Y_masks[i])
            Y_masks_val.append(Y_mask_val.to(device)*Y_masks[i])
                
        if not os.path.exists (output_path + sample_index_file): #'flux_split_year_v1.sav'
            torch.save({'sample_indexes':sample_indexes,
                    },output_path + sample_index_file) #'flux_split_year_v1.sav'
            

        ############################## finetune flux
        #load orginal model
        model_version= pretrained_model #'recotest_v11_exp4.sav_step4'  
        path_save = output_path+model_version
        checkpoint=torch.load(path_save)
        model1=kgml_lib.RecoGRU_multitask_v11_3(n_f,0.2,Tx,X_scaler[8,:],\
                                                Y1_scaler[0,:],Y2_scaler[0,:],Res_scaler[0,:])
        model1.load_state_dict(checkpoint['model_state_dict'])
        model1.to(device)
        path_save = output_path+ output_model #'recotest_v11_exp4_sample.sav_step5'
        
        ##freeze yield
        for name, param in model1.named_parameters():
            param.requires_grad = True
            if param.requires_grad and ('attn' in name or 'gru_basic' in name or 'densor_yield' in name):
                param.requires_grad = False
            if param.requires_grad:
                print(name)
        #initials
        loss_val_best = 500000
        R2_best=0.5
        best_epoch = 1500
        #train the model and pring loss/ yearly training
        lr_adam=0.001 #orginal 0.0001
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
        train_n = len(X_sites)
        #during training
        slw=365
        slw05=365
        train_losses = []
        val_losses = []
        maxepoch=30*4
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
                Y_Reco_obs_new.append(Y_Reco_obs[shuffled_b[i]])
                Y_NEE_obs_new.append(Y_NEE_obs[shuffled_b[i]])
                Y_masks_train_new.append(Y_masks_train[shuffled_b[i]])
                X_sites_new.append(X_sites[shuffled_b[i]])
            Y_Ra_pred_all=[]
            Y_Rh_pred_all=[]
            Y_Reco_pred_all=[]
            Y_NEE_pred_all=[]
            model1.zero_grad()
            ###keep Rh responses
            if epoch > 0:
                for feature in range(len(rangess)):
                    if feature == 7:
                        interval = intervals[feature]
                    else:
                        interval = intervals[feature]*intv_f
                    loss4 = check_Ra_Rh_response_v2(model1, check_xset, feature, rangess[feature], interval,\
                                                            Ra_r_mins[feature], Ra_daily_orgs[feature],\
                                                            Rh_r_mins[feature], Rh_daily_orgs[feature],Y1_scaler, device)
                    optimizer.zero_grad()
                    loss4.backward()
                    optimizer.step()  
                    with torch.no_grad():
                        train_loss4=train_loss4+loss4.item()
            for bb in range(train_n):
                totsq = X_sites_new[bb].size(1)
                maxit = int((totsq-slw)/slw05+1)
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
                Ra_temp = -1.0*Z_norm_reverse(Y1_pred_sq[0,:,0],Y1_scaler[0,:],1.0).view(-1)
                Rh_temp = -1.0*Z_norm_reverse(Y1_pred_sq[0,:,1],Y1_scaler[1,:],1.0).view(-1)
                Y_Ra_pred_all.append(Ra_temp)
                Y_Rh_pred_all.append(Rh_temp)
                Y_Reco_pred_all.append(Ra_temp+Rh_temp)
                Y_NEE_pred_all.append(Z_norm_reverse(Y1_pred_sq[0,:,2],Y1_scaler[2,:],1.0).view(-1))
                for zz in range(len(hidden)):
                    hidden[zz].detach_()  
            #######MSE and mass balance loss
            loss,loss1,loss2,loss3 = myloss_mb_flux_mask_re_v2(Y_Ra_pred_all,Y_Rh_pred_all,Y_NEE_pred_all,\
                                                    Y_Reco_obs_new, Y_NEE_obs_new,\
                                    X_sites_new,X_scaler[8,:], Y1_scaler,\
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
                train_loss=[train_loss,train_loss1/train_n,train_loss2/train_n,train_loss3/train_n,train_loss4/n_f]
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
                            Reco_obs_masked.contiguous().view(-1)).item(),\
                        compute_r2(NEE_pred_masked.contiguous().view(-1),
                            NEE_obs_masked.contiguous().view(-1)).item()]

                Y_Ra_pred_all=[]
                Y_Rh_pred_all=[]
                Y_Reco_pred_all=[]
                Y_NEE_pred_all=[]
                for bb in range(train_n):
                    totsq = X_sites[bb].size(1)
                    maxit = int((totsq-slw)/slw05+1)
                    hidden = model1.init_hidden(1)
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(X_sites[bb][:,slw05*it:slw05*it+slw,:],hidden)
                    Y1_pred_sq = torch.zeros([1,totsq,3],device=device)
                    for it in range(maxit):
                        #change intitial for year round simulation
                        Y1_pred,Y2_pred, hidden = model1(X_sites[bb][:,slw05*it:slw05*it+slw,:],hidden)
                        Y1_pred_sq[:,slw05*it:slw05*it+slw,:] = Y1_pred[:,:,:]
                    Y_Reco_pred_all.append(-1.0*(Z_norm_reverse(Y1_pred_sq[0,:,0],Y1_scaler[0,:],1.0)+\
                                                Z_norm_reverse(Y1_pred_sq[0,:,1],Y1_scaler[1,:],1.0)).view(-1))
                    Y_NEE_pred_all.append(Z_norm_reverse(Y1_pred_sq[0,:,2],Y1_scaler[2,:],1.0).view(-1))
                loss,loss1,loss2,loss3 = myloss_mb_flux_mask_re(Y_Reco_pred_all,Y_NEE_pred_all,Y_Reco_obs, Y_NEE_obs,\
                                    X_sites, X_scaler[8,:], Y1_scaler,\
                                    Y_masks_val, lamda,tol_mb)
                #calculate response to SOC:
                loss4 = 0.0
                for feature in range(len(rangess)):
                    if feature == 7:
                        interval = intervals[feature]
                    else:
                        interval = intervals[feature]*intv_f
                    loss4_t = check_Ra_Rh_response_v2(model1, check_xset, feature, rangess[feature], interval,\
                                                            Ra_r_mins[feature], Ra_daily_orgs[feature],\
                                                            Rh_r_mins[feature], Rh_daily_orgs[feature],Y1_scaler, device)
                    loss4 = loss4+loss4_t
                val_loss=[loss.item(),loss1.item()/train_n,loss2.item()/train_n,loss3.item()/train_n,loss4.item()/n_f]
                val_losses.append(val_loss)
                #R2 for all
                Y_masksb =[]
                for i in range(train_n):
                    Y_masksb.append(Y_masks_val[i].ge(0.5))
                Y_masksb = torch.cat(Y_masksb,dim=0)
                
                Reco_pred_masked = torch.masked_select(torch.cat(Y_Reco_pred_all,dim=0),Y_masksb)
                Reco_obs_masked = torch.masked_select(torch.cat(Y_Reco_obs,dim=0),Y_masksb)
                NEE_pred_masked = torch.masked_select(torch.cat(Y_NEE_pred_all,dim=0),Y_masksb)
                NEE_obs_masked = torch.masked_select(torch.cat(Y_NEE_obs,dim=0),Y_masksb)
                val_R2 = [compute_r2(Reco_pred_masked.contiguous().view(-1),
                            Reco_obs_masked.contiguous().view(-1)).item(),\
                        compute_r2(NEE_pred_masked.contiguous().view(-1),
                            NEE_obs_masked.contiguous().view(-1)).item()]
                #if val_loss < loss_val_best and val_R2 > R2_best:
                if val_loss[0] < loss_val_best:
                    loss_val_best=val_loss[0]
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

                if np.mean(train_R2)> 0.999 or (epoch - best_epoch) > 30:
                    break
            model1.train()
        endtime=time.time()
        path_fs = path_save+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model1.state_dict(),
                    }, path_fs)
        print("Step 5 saved model is: ", path_fs)
        print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best)
        print(f"total Training time: {endtime - starttime}s")