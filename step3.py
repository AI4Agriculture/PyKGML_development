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
import copy
####import kgml_lib
import kgml_lib

def train(data_path:str, output_path:str, input_data:str,
          pretrained_model:str, output_model:str, synthetic_data:str): #sample_index_file:str, 
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
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
    lamda = [1.0,1.0]
    tol_mb = 0.01

    #####define functions from kgml_lib
    Z_norm = kgml_lib.Z_norm 
    Z_norm_reverse = kgml_lib.Z_norm_reverse
    Z_norm_with_scaler = kgml_lib.Z_norm_with_scaler
    get_gpu_memory = kgml_lib.get_gpu_memory
    compute_r2=kgml_lib.R2Loss()
    Check_yield_GPP = kgml_lib.Check_yield_GPP
    check_yield_response_loss = kgml_lib.check_yield_response_loss

    ####load the county basic data
    data_dir = data_path+'combine_random_sample_size_300/'
    sample_size = 300
    #load county ID
    county_FIPS = np.load(data_dir+'county_FIPS.npy').tolist()
    print(len(county_FIPS))
    dropsites = [38039,27123,47095,47183,47053,47045,47079] # remove the county without Corn/Soybean rotation fields
    for s in dropsites:
        county_FIPS.remove(s)
    print(len(county_FIPS))
    county_FIPS = np.array(county_FIPS)
    #load corn and soybean fraction (Dictionary)
    corn_fraction = np.load(data_dir+'corn_fraction_sample_%d.npy'  % (sample_size),allow_pickle=True).item()
    soybean_fraction = np.load(data_dir+'soybean_fraction_sample_%d.npy'  % (sample_size),allow_pickle=True).item()
    print(corn_fraction[county_FIPS[1]].shape)
    #load observed crop yield (Dictionary)
    obs_corn_yield = np.load(data_dir+'obs_corn_yield.npy',allow_pickle=True).item()
    obs_soybean_yield = np.load(data_dir+'obs_soybean_yield.npy',allow_pickle=True).item()
    print(obs_corn_yield[county_FIPS[1]].shape)
    convert_index_corn = obs_corn_yield['gC/m2/year_to_Bu/Acre'] # the coeeficent to convert gC/m2/year to Bu/Acre for corn
    convert_index_soybean = obs_soybean_yield['gC/m2/year_to_Bu/Acre'] # the coeeficent to convert gC/m2/year to Bu/Acre for soybean
    print(convert_index_corn)
    get_gpu_memory()
    tot_n = len(county_FIPS)
    #separate training/val 420/210
    shuffled_b=torch.randperm(tot_n)
    if not os.path.exists (output_path + 'yield_split_v2.sav'):
        torch.save({'shuffled_b':shuffled_b,
                    },output_path + 'yield_split_v2.sav')
    else:
        tmp = torch.load(output_path + 'yield_split_v2.sav')
        shuffled_b = tmp['shuffled_b']
    county_FIPS_val = county_FIPS[shuffled_b[210:310]]
    county_FIPS_train = county_FIPS[shuffled_b[310:tot_n]]
    ##load synthetic data for GPP response calculations
    tmp = torch.load(output_path + synthetic_data) ###only for test #'sys_data1.sav'
    check_xset = tmp['sys_data1']
    features = [8,9,16]
    rangess = [[-2, 2],[-3.5, 2.5],[-2.5, 5],[-1.5,3.5],[-1.5,7],[-2.0,5.5],[-0.3889,10.0],[1,5],[-0.5,5],[-1.6384,1.6384],
            [-10,2],[-0.75,4],[-3.25,1.25],[-4.5,5.0],[-2.5,3],[-0.75,7.75],[-0.5,5],[-3.25,4.25],[-1.75,3.5]]
    intervals = [0.2,0.3,0.5,0.25,0.5,0.5,0.5,2,0.25,1.6384*2/17,0.5,0.25,0.25,0.5,0.25,0.5,0.25,0.5,0.25]

    #####here we only use 2 county for training and 2 county for validation as an example. 
    #####to run full training, you may need to use all counties
    county_FIPS_val = county_FIPS_val[0:2]
    county_FIPS_train = county_FIPS_train[0:2]
    print(county_FIPS_val,county_FIPS_train)

    #####Train with all data!!
    starttime=time.time()
    for cv_n in range(0,1):
        tyear = 21
        Tx = 365
        val_n = len(county_FIPS_val)
        train_n = len(county_FIPS_train)
        # for load in the observed yield
        Y_obs_train = torch.zeros([train_n,tyear,2])
        Y_obs_val = torch.zeros([val_n,tyear,2])
        Y_obs_mask_train = torch.zeros([train_n,tyear,2])+1.0
        Y_obs_mask_val = torch.zeros([val_n,tyear,2])+1.0
        # for load in the fraction
        Y_maskb_train = []
        Y_maskb_val = []
        # for load in X_input
        X_input_train = []
        X_input_val = []
        for i in range(len(county_FIPS_train)):
            fips = county_FIPS_train[i]
            Y_obs_train[i,:,0] = Z_norm_with_scaler(torch.from_numpy(obs_corn_yield[fips]).view(-1),Y2_scaler[0,:])
            Y_obs_train[i,:,1] = Z_norm_with_scaler(torch.from_numpy(obs_soybean_yield[fips]).view(-1),Y2_scaler[0,:])
            X_input_train.append(torch.from_numpy(np.load(data_dir+'Pred_xset_%d_sample_%d.npy.npz'  % (fips,sample_size))['arr_0']).float())
            Y_maskb_train.append([torch.from_numpy(corn_fraction[fips]).float().ge(0.5).to(device),torch.from_numpy(soybean_fraction[fips]).float().ge(0.5).to(device)])
            if torch.isnan(torch.mean(X_input_train[i])):
                print('!!!!!!!!!!!!X_input has nan', i,fips, X_input_train[i].size(0),\
                        len(torch.nonzero(torch.isnan(torch.mean(torch.mean(X_input_train[i],dim=2),dim=1)))))
                print("index",torch.isnan(torch.mean(torch.mean(X_input_train[i],dim=2),dim=1)).nonzero(as_tuple=True))
                    #directly mask out nan point
                nanindex_dim0 = torch.isnan(torch.mean(torch.mean(X_input_train[i],dim=2),dim=1))
                X_input_train[i][nanindex_dim0,:,:] = 0.01
                Y_maskb_train[i][0][nanindex_dim0,:] = False
                Y_maskb_train[i][1][nanindex_dim0,:] = False
            print("Train finished: ", i, fips, corn_fraction[fips].shape,np.count_nonzero(np.isnan(obs_corn_yield[fips]+obs_corn_yield[fips])),\
                np.count_nonzero(np.isnan(corn_fraction[fips]+soybean_fraction[fips]))) 
                
        for i in range(len(county_FIPS_val)):
            fips = county_FIPS_val[i]
            Y_obs_val[i,:,0] = Z_norm_with_scaler(torch.from_numpy(obs_corn_yield[fips]).view(-1),Y2_scaler[0,:])#scale the observed data
            Y_obs_val[i,:,1] = Z_norm_with_scaler(torch.from_numpy(obs_soybean_yield[fips]).view(-1),Y2_scaler[0,:])
            Y_maskb_val.append([torch.from_numpy(corn_fraction[fips]).float().ge(0.5).to(device),torch.from_numpy(soybean_fraction[fips]).float().ge(0.5).to(device)])
                ### real data may not have 300 sample size, input data may also has nan
            X_input_val.append(torch.from_numpy(np.load(data_dir+'Pred_xset_%d_sample_%d.npy.npz'  % (fips,sample_size))['arr_0']).float()) 
            if torch.isnan(torch.mean(X_input_val[i])):
                print('!!!!!!!!!!!!X_input has nan', i,fips, X_input_val[i].size(0),\
                        len(torch.nonzero(torch.isnan(torch.mean(torch.mean(X_input_val[i],dim=2),dim=1)))))
                print("index",torch.isnan(torch.mean(torch.mean(X_input_val[i],dim=2),dim=1)).nonzero(as_tuple=True))
                nanindex_dim0 = torch.isnan(torch.mean(torch.mean(X_input_val[i],dim=2),dim=1))
                X_input_val[i][nanindex_dim0,:,:] = 0.01
                Y_maskb_val[i][0][nanindex_dim0,:] = False
                Y_maskb_val[i][1][nanindex_dim0,:] = False
            print("Val finished: ", i, fips, corn_fraction[fips].shape,np.count_nonzero(np.isnan(obs_corn_yield[fips]+obs_corn_yield[fips])),\
                np.count_nonzero(np.isnan(corn_fraction[fips]+soybean_fraction[fips])))
            
        #remove nan
        nanindex = torch.isnan(Y_obs_train)
        Y_obs_train[nanindex] = 0.0
        Y_obs_mask_train[nanindex] = 0.0
        nanindex = torch.isnan(Y_obs_val)
        Y_obs_val[nanindex] = 0.0
        Y_obs_mask_val[nanindex] = 0.0
        Y_obs_train = Y_obs_train.to(device)
        Y_obs_val = Y_obs_val.to(device)
        Y_obs_mask_train = Y_obs_mask_train.to(device)
        Y_obs_mask_val = Y_obs_mask_val.to(device)
        # print(Y_obs_train,Y_obs_mask_train)
        #  break
        
        ###test
            #train the model and pring loss
        ##############################1) train yield 
            ##freeze Rh and NEE and Ra
        #load orginal model
        model_version= pretrained_model #'recotest_v11_exp4.sav_step2'
        path_save = output_path+model_version
        checkpoint=torch.load(path_save)
        seq_len = 365
        model1=kgml_lib.RecoGRU_multitask_v11_3(n_f,0.2,seq_len,X_scaler[8,:],Y1_scaler[0,:],Y2_scaler[0,:],Res_scaler[0,:])
        model1.load_state_dict(checkpoint['model_state_dict'])
        model1.to(device) 
        path_save = output_path+ output_model #'recotest_v11_exp4_sample.sav_step3'
        for name, param in model1.named_parameters():
            param.requires_grad = True
            if param.requires_grad and ('gru_Rh' in name or 'densor_Rh' in name or 'gru_Ra' in name or 'densor_Ra' in name \
                                    or 'gru_NEE' in name or 'densor_NEE' in name):
                param.requires_grad = False
            if param.requires_grad:
                print(name)
        #initials
        loss_val_best = 500000
        R2_best=0.5
        best_epoch = 1000
        SPL_yield = kgml_lib.SPLoss_Yield_v2(1,1.3) ####if use all data, 0.1,1.3

        #train the model and pring loss/ yearly training
        lr_adam=0.001 #orginal 0.001
        optimizer = optim.Adam([
                    {'params': model1.gru_Rh.parameters()},
                    {'params': model1.densor_Rh.parameters()},
                    {'params': model1.gru_NEE.parameters()},
                    {'params': model1.densor_NEE.parameters()},
                    {'params': model1.gru_Ra.parameters()},
                    {'params': model1.densor_Ra.parameters()},
                    {'params': model1.attn.parameters()},
                    {'params': model1.densor_yield.parameters()},
                    {'params': model1.gru_basic.parameters(), 'lr': lr_adam*0.2}
                ], lr=lr_adam) #add weight decay normally 1-9e-4
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # bsz = 21 ##here batch must can divide the train_n
        bsz = 2 ###sample
        totsq=Tx*tyear
        #during training
        slw=365
        slw05=365
        maxit=int((totsq-slw)/slw05+1)
        train_losses = []
        val_losses = []
        maxepoch=40
        model1.train()
        for epoch in range(maxepoch):
            train_loss1=0.0
            train_loss2=0.0
            train_loss3=0.0
            val_loss1=0.0
            val_loss2=0.0
            val_loss3=0.0
            #shuffled the training data
            shuffled_b=torch.randperm(train_n)
            county_FIPS_train_new=county_FIPS_train[shuffled_b]
            Y_obs_train_new=Y_obs_train[shuffled_b,:,:]
            Y_obs_mask_train_new = Y_obs_mask_train[shuffled_b,:,:]
            Y_maskb_train_new=[]
            shuffled_b1s = []
            for i in range(train_n):
                #to shuffle 300 points
                shuffled_b1 = torch.randperm(Y_maskb_train[shuffled_b[i]][0].size(0))
                if len(shuffled_b1) > 200:
                    shuffled_b1 = shuffled_b1[0:200]
                shuffled_b1s.append(shuffled_b1)    
                Y_maskb_train_new.append([Y_maskb_train[shuffled_b[i]][0][shuffled_b1,:],\
                                        Y_maskb_train[shuffled_b[i]][1][shuffled_b1,:]])
        
            Y_pred_all = torch.zeros([train_n,tyear,2],device = device)
            model1.zero_grad()
            val_flag = False
            for bb in range(int(train_n/bsz)):
                if bb != int(train_n/bsz)-1:
                    sbb = bb*bsz
                    ebb = (bb+1)*bsz
                else:
                    sbb = bb*bsz
                    ebb = train_n
                hiddens = []
                for it in range(maxit):
                    Y_pred_batch = torch.zeros([ebb-sbb,2],device = device)
                    ###load in X_input data for this batch
                    for i in range(sbb,ebb):
                        X_input =  X_input_train[shuffled_b[i]][shuffled_b1s[i],:,:].to(device)  
                        if it == 0:
                            #model initials
                            with torch.no_grad():
                                hidden = model1.init_hidden(X_input.size(0))
                                ##spinup 2 years for turnover
                                for itt in range(2):
                                    __,___,hidden = model1(X_input[:,slw05*itt:slw05*itt+slw,:],hidden) 
                                hiddens.append(hidden)
                        __,Y2_pred,hidden = model1(X_input[:,slw05*it:slw05*it+slw,:],hiddens[i-sbb])
                        for zz in range(len(hidden)):
                            hidden[zz].detach_()
                        with torch.no_grad():
                            hiddens[i-sbb] = hidden
                        #add check nan for corn
                        Y2_pred_masked = torch.masked_select(Y2_pred.view(-1),Y_maskb_train_new[i][0][:,it].view(-1).to(device))
                        if len(Y2_pred_masked) !=0:
                            Y_pred_batch[i-sbb,0] = torch.mean(Y2_pred_masked) ##for corn
                        else:
                            Y_pred_batch[i-sbb,0] = 0.0
                            Y_obs_mask_train_new[i,it,0] = 0.0
                        #add check nan for soybean
                        Y2_pred_masked = torch.masked_select(Y2_pred.view(-1),Y_maskb_train_new[i][1][:,it].view(-1).to(device))
                        if len(Y2_pred_masked) !=0:
                            Y_pred_batch[i-sbb,1] = torch.mean(Y2_pred_masked) ##for soybean
                        else:
                            Y_pred_batch[i-sbb,1] = 0.0
                            Y_obs_mask_train_new[i,it,1] = 0.0
                        ####make shure 0< yield < 0.5*GPP
                        GPP = torch.sum(Z_norm_reverse(X_input[:,slw05*it:slw05*it+slw,8],X_scaler[8,:]),dim=1).view(-1).to(device)
                        Yield = Z_norm_reverse(Y2_pred, Y2_scaler[0,:]).view(-1)
                        if i == sbb:
                            loss3 = Check_yield_GPP(Yield,GPP)
                        else:
                            loss3 = loss3 + Check_yield_GPP(Yield,GPP)
                    loss3 = loss3/(ebb-sbb)
                    loss1 = SPL_yield(Y_pred_batch[:,0].view(-1), Y_obs_train_new[sbb:ebb,it,0].view(-1), \
                                    Y_pred_batch[:,1].view(-1), Y_obs_train_new[sbb:ebb,it,1].view(-1), \
                                    Y_obs_mask_train_new[sbb:ebb,it,0].view(-1),\
                                    Y_obs_mask_train_new[sbb:ebb,it,1].view(-1),\
                                    val_flag,device)
                    loss2 = 0.0
                    for feature in features:
                        loss2 = loss2 + check_yield_response_loss(model1,check_xset, feature, rangess[feature],\
                                                                intervals[feature],device)
                    loss = loss1+loss2+loss3
                    optimizer.zero_grad()
                    if loss.item() != 0.0:
                        loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        train_loss1=train_loss1+loss1.item()
                        train_loss2=train_loss2+loss2.item()
                        train_loss3=train_loss3+loss3.item()
                        Y_pred_all[sbb:ebb,it,:]=Y_pred_batch
            scheduler.step()
            get_gpu_memory()
            #update self-paced loss threshold
            if epoch < 100:
                SPL_yield.increase_threshold()
            #validation
            model1.eval()
            val_flag = True
            with torch.no_grad():
                train_loss1=train_loss1/(maxit*train_n/bsz)
                train_loss2=train_loss2/(maxit*train_n/bsz)
                train_loss3=train_loss3/(maxit*train_n/bsz)
                train_loss = [train_loss1,train_loss2,train_loss3]
                train_losses.append(train_loss)
                #R2 for all
                maskbt = Y_obs_mask_train_new.ge(0.5)
                Y_obs_train_new_maksed = torch.masked_select(Y_obs_train_new,maskbt)
                Y_pred_all_maksed = torch.masked_select(Y_pred_all,maskbt)
                train_R2 = compute_r2(Y_pred_all_maksed.contiguous().view(-1),Y_obs_train_new_maksed.contiguous().view(-1)).item()
                
                Y_pred_all=torch.zeros([val_n,tyear,2],device = device)
                for bb in range(val_n):
                    hidden = model1.init_hidden(X_input_val[bb].size(0))
                    ##spinup 2 years for turnover
                    for it in range(2):
                        __,___,hidden = model1(X_input_val[bb][:,slw05*it:slw05*it+slw,:].to(device),hidden)
                    for it in range(maxit):
                            #change intitial for year round simulation
                        __,Y2_pred, hidden = model1(X_input_val[bb][:,slw05*it:slw05*it+slw,:].to(device),hidden)
                        #add check nan for corn
                        Y2_pred_masked = torch.masked_select(Y2_pred.view(-1),Y_maskb_val[bb][0][:,it].view(-1).to(device))
                        if len(Y2_pred_masked) !=0:
                            Y_pred_all[bb,it,0] = torch.mean(Y2_pred_masked) ##for corn
                        else:
                            Y_pred_all[bb,it,0] = 0.0
                            Y_obs_mask_val[bb,it,0] = 0.0
                        #add check nan for soybean
                        Y2_pred_masked = torch.masked_select(Y2_pred.view(-1),Y_maskb_val[bb][1][:,it].view(-1).to(device))
                        if len(Y2_pred_masked) !=0:
                            Y_pred_all[bb,it,1] = torch.mean(Y2_pred_masked) ##for soybean
                        else:
                            Y_pred_all[bb,it,1] = 0.0
                            Y_obs_mask_val[bb,it,1] = 0.0
                        ####make shure 0< yield < 0.5*GPP
                        GPP = torch.sum(Z_norm_reverse(X_input_val[bb][:,slw05*it:slw05*it+slw,8],X_scaler[8,:]),dim=1).view(-1).to(device)
                        Yield = Z_norm_reverse(Y2_pred, Y2_scaler[0,:]).view(-1)
                        if bb == 0 and it==0:
                            loss3 = Check_yield_GPP(Yield,GPP)
                        else:
                            loss3 = loss3 + Check_yield_GPP(Yield,GPP)
                loss3 = loss3/(maxit*val_n)
                
                loss1 = SPL_yield(Y_pred_all[:,:,0], Y_obs_val[:,:,0], Y_pred_all[:,:,1], Y_obs_val[:,:,1], \
                                Y_obs_mask_val[:,:,0],Y_obs_mask_val[:,:,1],\
                                val_flag,device)
                loss2 = 0.0
                for feature in features:
                    loss2 = loss2 + check_yield_response_loss(model1,check_xset, feature,\
                                                            rangess[feature], intervals[feature],device)
                val_loss = [loss1.item(),loss2.item(),loss3.item()]
                val_losses.append([loss1.item(),loss2.item(),loss3.item()])
                #R2 for all
                maskbt = Y_obs_mask_val.ge(0.5)
                Y_obs_val_maksed = torch.masked_select(Y_obs_val,maskbt)
                Y_pred_all_maksed = torch.masked_select(Y_pred_all,maskbt)
                val_R2 = compute_r2(Y_pred_all_maksed.contiguous().view(-1),Y_obs_val_maksed.contiguous().view(-1)).item()
                #if val_loss < loss_val_best and val_R2 > R2_best:
                if val_loss[0]+val_loss[1]+val_loss[2] < loss_val_best:
                    loss_val_best=val_loss[0]+val_loss[1]+val_loss[2]
                    R2_best = val_R2
                    best_epoch = epoch
                    #os.remove(path_save)
                    if epoch < 20:
                        torch.save({'epoch': epoch,
                                'model_state_dict': model1.state_dict(),
                                'R2': train_R2,
                                'loss': train_loss,
                                'los_val': val_loss,
                                'R2_val': val_R2,
                                }, path_save) 
                    else:
                        torch.save({'epoch': epoch,
                                'model_state_dict': model1.state_dict(),
                                'R2': train_R2,
                                'loss': train_loss,
                                'los_val': val_loss,
                                'R2_val': val_R2,
                                }, path_save+'_1') 
                print("finished training epoch", epoch+1)
                
                mtime=time.time()
                print("train_loss: ", train_loss, "train_R2", train_R2,"val_loss:",val_loss,"val_R2", val_R2,\
                    "loss val best:",loss_val_best,"R2 val best:",R2_best, f"Spending time: {mtime - starttime}s")
                #save final every time
                path_fs = path_save+'fs'
                torch.save({'train_losses': train_losses,
                            'val_losses': val_losses,
                            'model_state_dict_fs': model1.state_dict(),
                            }, path_fs)  
                if np.mean(train_R2)> 0.999 or (epoch - best_epoch) > 10:
                    break
            model1.train()
        endtime=time.time()

        print("Step 3 saved model is: ", path_fs)
        print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best)
        print(f"total Training time: {endtime - starttime}s")