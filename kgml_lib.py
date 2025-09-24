##############################################################################
## Copyright (C) 2023 - All Rights Reserved
## This file is part of the manuscript named "Knowledge-based artificial 
## intelligence significantly improved agroecosystem carbon cycle 
## quantification". Unauthorized copying/distributing/modifying of this file, 
## via any medium is strictly prohibited.
## Proprietary and confidential Written by DBRP authors of above manuscript
#############################################################################
import numpy as np
import pandas as pd
import math
from io import open
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess as sp
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

##########################################################################################
#basic functions
#
##########################################################################################
class R2Loss(nn.Module):
    #calculate coefficient of determination r2
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y

class R2Loss_featurewise(nn.Module):
    def forward(self, y_pred, y):
        # Compute R2 per feature across the sequence dimension
        var_y = torch.var(y, dim=1, unbiased=False)  # Shape: [N, feature]
        mse_loss = F.mse_loss(y_pred, y, reduction="none")  # Shape: [N, sequence, feature]
        mse_loss = torch.mean(mse_loss, dim=1)  # Average over the sequence dimension, shape: [N, feature]

        r2 = 1.0 - (mse_loss / var_y)  # Compute R^2 per feature
        return torch.mean(r2, dim=0)


def multiTaskWeighted_Loss(output, target, loss_weights):
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
    
def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

def Z_norm(X):
    X_mean=X.numpy().mean(dtype=np.float32)
    X_std=np.std(np.array(X,dtype=np.float32))
    return (X-X_mean)/X_std, X_mean, X_std

def Z_norm_reverse(X,Xscaler,units_convert=1.0):
    return (X*Xscaler[1]+Xscaler[0])*units_convert

def scalar_maxmin(X):
    return (X - X.min())/(X.max() - X.min())

def Z_norm_with_scaler(X,Xscaler):
    return (X-Xscaler[0])/Xscaler[1]

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

#sample data considering dropout and leadtime (120 days from 365 days)    
def sample_data_FN(X, Y, fn_ind):
    #find the fertilized time
    # print(np.sum(X[:,1,fn_ind].to("cpu").numpy()>0))
    fntime_ind=np.where(X[:,1,fn_ind].view(-1).to("cpu").numpy()>0)[0]
    # print(fntime_ind)
    #get focused data only for fertilized period with random leading time
    for t in fntime_ind:
        if t == fntime_ind[0]:
            X_new = X[t-30:t+90,:,:]
            Y_new = Y[t-30:t+90,:,:]
        else:
            X_new = torch.cat((X_new,X[t-30:t+90,:,:]),1)
            Y_new = torch.cat((Y_new,Y[t-30:t+90,:,:]),1)
    return X_new,Y_new

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
#for multi-task learning, sumloss
def myloss_mul_sum(output, target,loss_weights):
    loss = 0.0
    nout=output.size(2)
    for i in range(nout):
        loss = loss + loss_weights[i]*torch.mean((output[:,:,i] - target[:,:,i])**2)
    return loss


def get_R2_score(target, prediction, y_scaler:list, output_feature_name:list):
    y_param_num = target.shape[-1]
    prediction_flat = prediction.reshape(-1,y_param_num)
    target_flat = target.reshape(-1,y_param_num)
    for i in range(y_param_num):
        _r2 = r2_score(Z_norm_reverse(target_flat[:,i], y_scaler[i]), Z_norm_reverse(prediction_flat[:,i], y_scaler[i]))
        print(f"Feature {output_feature_name[i]} R2 Score is: {_r2}")



##########################################################################################
#plotting functions
#
##########################################################################################
def plot_features(data, feature_number, feature_name, sub_title):
    n_features = feature_number 
    
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)
    
    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    # plt.figure(figsize=(15, 10))
    # plt.title('Scaled Input features')
    # Loop through each feature and create a histogram subplot
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i+1)  # Adjust grid (3 rows, 4 columns) as needed
        plt.hist(data[:, i], bins=30, edgecolor='black', alpha=0.7)
        # plt.title(f'Feature {i+1}')
        _f_name = feature_name[i] 
        plt.title(_f_name)
        # plt.xlabel(_f_name)
        plt.ylabel('Frequency')
    
    # plt.tight_layout()
    # Add a main title for all subplots
    # Set a suptitle with a lower y value to bring it closer to subplots
    # positions the overall title at 92% of the figure height (you can try lowering this value further if needed).
    # plt.suptitle("Distribution of Scaled Features", fontsize=16, y=0.95)
    plt.suptitle(sub_title, fontsize=16)
    
    # Adjust subplots: increase the 'top' value to reduce the gap between the title and subplots.
    # plt.subplots_adjust(top=0.92, hspace=0.5)
    
    # Adjust the vertical space between rows (hspace)
    # plt.subplots_adjust(hspace=0.4)  # Increase or decrease 0.5 as needed

    # Reserve space at the top (e.g., 5% margin) so the suptitle doesn't overlap
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_result(y_scaler:list, features:list, all_predictions_flat:list,all_targets_flat:list, site:int, sub_title:str=None):

    N, F = all_targets_flat.shape # N: 365, F: features number

    fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

    for i, name in enumerate(features):
        ax_line = axes[i]

        if y_scaler:
            y_true = Z_norm_reverse(all_targets_flat[:,i], y_scaler[i])
            y_pred = Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i])
        else:
            y_true = all_targets_flat[:,i]
            y_pred = all_predictions_flat[:,i]

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
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
        full_title = f"Sample {site}"
    else:
        full_title = f"{sub_title} Sample {site}"
    fig.suptitle(full_title, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.show()

def scatter_result(y_scaler:list, features:list, all_predictions_flat, all_targets_flat,sub_title:str=None):

    N, F = all_targets_flat.shape # N: 365, F: features number

    fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

    for i, name in enumerate(features):
        ax_scatter = axes[i]
        if y_scaler:
            y_true = Z_norm_reverse(all_targets_flat[:,i], y_scaler[i])
            y_pred = Z_norm_reverse(all_predictions_flat[:,i], y_scaler[i])
        else:
            y_true = all_targets_flat[:,i]
            y_pred = all_predictions_flat[:,i]

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
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

# Scatter the prediction value and true values base on test dataset
def vis_scatter_prediction_result(target, prediction, y_scaler:list, features:list):
    y_param_num = target.shape[-1]
    # self.all_predictions shape is [200, 365, features]

    prediction_flat = prediction.reshape(-1,y_param_num)
    target_flat     = target.reshape(-1,y_param_num)
    scatter_result(y_scaler, features, prediction_flat, target_flat)
    

def vis_plot_prediction_result_time_series(target, prediction, y_scaler:list, features:list, sample:int):
    y_param_num = target.shape[-1]
    # self.all_predictions shape is [200, 365, features]
    _idx = sample
    all_predictions_flat = prediction[_idx, :, :].reshape(-1, y_param_num)
    all_targets_flat     = target[_idx, :, :].reshape(-1, y_param_num)
    plot_result(y_scaler, features, all_predictions_flat, all_targets_flat, sample)


def vis_plot_prediction_result_time_series_masked(target, prediction, y_scaler:list, features:list, sample:int, obs_mask:np.ndarray=None):
    y_param_num = target.shape[-1]
    # self.all_predictions shape is [200, 365, features]
    _idx = sample
    prediction_flat = prediction[_idx, :, :].reshape(-1,y_param_num)
    target_flat     = target[_idx,:,:].reshape(-1,y_param_num)
    mask = obs_mask.reshape(target.shape)[_idx,:,:].reshape(-1,y_param_num)
    N, F = target_flat.shape # N: 365, F: features number

    fig, axes = plt.subplots(F, 1, figsize=(12, 4*F))

    for i, name in enumerate(features):
        ax_line = axes[i]

        y_true = Z_norm_reverse(target_flat[:,i], y_scaler[i]).numpy()
        y_pred = Z_norm_reverse(prediction_flat[:,i], y_scaler[i]).numpy()
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
    sub_title = f"Sample {sample}"
    fig.suptitle(sub_title, fontsize=12)
    fig.subplots_adjust(top=0.9, hspace=0.4)
    plt.show()


##########################################################################################
#loss functions
#
##########################################################################################
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
#for multi-task learning, sumloss
def myloss_mul_sum(output, target,loss_weights):
    loss = 0.0
    nout=output.size(2)
    for i in range(nout):
        loss = loss + loss_weights[i]*torch.mean((output[:,:,i] - target[:,:,i])**2)
    return loss
    
class SPLoss_YieldRa(nn.Module):
    def __init__(self,threshold, growing_factor):
        super(SPLoss_YieldRa, self).__init__()
        self.threshold = threshold
        self.growing_factor = growing_factor
        
    def forward(self, output1, target1,output2, target2, val_flag):
        #Ra loss
        super_loss1 = torch.mean((output1[:,:] - target1[:,:])**2,dim=1)
        #Yield loss
        super_loss2 = torch.mean((output2[:,:] - target2[:,:])**2,dim=1)
        if val_flag:
            loss1 = torch.mean(super_loss1)
            loss2 = torch.mean(super_loss2)
        else:
            v1 = self.spl_loss(super_loss1)
            v2 = self.spl_loss(super_loss2)
            loss1 = torch.mean(super_loss1 * v1)
            loss2 = torch.mean(super_loss2 * v2)
        loss = loss1 + loss2
        return loss,loss1,loss2
        
    def increase_threshold(self):
        self.threshold *= self.growing_factor
        
    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.float() 
    
#for multi-task learning, sumloss
#lamda is the weights if each lossl tol_MB is the tolerance of mass balance
ReLU1 = nn.ReLU()
def myloss_mb_flux(output1, target1, GPP, GPP_scaler, Y1_scaler,loss_weights1, lamda,tol_MB):
    #Ra, Rh and NEE loss
    loss1 = 0.0
    nout1=output1.size(2)
    for i in range(nout1):
        loss1 = loss1 + loss_weights1[i]*torch.mean((output1[:,:,i] - target1[:,:,i])**2)
    #mass balance loss GRR+Ra+Rh-NEE = 0   (ra, rh are negative)
    GPP = Z_norm_reverse(GPP,GPP_scaler,1.0)
    Ra = Z_norm_reverse(output1[:,:,0],Y1_scaler[0,:],1.0)
    Rh = Z_norm_reverse(output1[:,:,1],Y1_scaler[1,:],1.0)
    NEE = Z_norm_reverse(output1[:,:,2],Y1_scaler[2,:],1.0)
    loss2 =  torch.mean(ReLU1(torch.abs(GPP + Ra + Rh+NEE) - tol_MB*torch.abs(Ra + Rh)))
    loss = lamda[0]*loss1 + lamda[1]*loss2
    return loss,loss1,loss2

#for multi-task learning, sumloss
#lamda is the weights if each lossl tol_MB is the tolerance of mass balance
ReLU1 = nn.ReLU()
def myloss_mb_flux_mask(output1, target1, GPP, GPP_scaler, Y1_scaler,Y1_mask, loss_weights1, lamda,tol_MB):
    #Ra, Rh and NEE loss
    loss1 = 0.0
    nout1=output1.size(2)
    for i in range(nout1):
        loss1 = loss1 + loss_weights1[i]*torch.mean(Y1_mask*(output1[:,:,i] - target1[:,:,i])**2)
    #mass balance loss GRR+Ra+Rh-NEE = 0   (ra, rh are negative)
    GPP = Z_norm_reverse(GPP,GPP_scaler,1.0)
    Ra = Z_norm_reverse(output1[:,:,0],Y1_scaler[0,:],1.0)
    Rh = Z_norm_reverse(output1[:,:,1],Y1_scaler[1,:],1.0)
    NEE = Z_norm_reverse(output1[:,:,2],Y1_scaler[2,:],1.0)
    loss2 =  torch.mean(Y1_mask*(ReLU1(torch.abs(GPP + Ra + Rh+NEE) - tol_MB*torch.abs(Ra + Rh))))
    loss = lamda[0]*loss1 + lamda[1]*loss2
    return loss,loss1,loss2


####changed due to packaging
def check_Rh2SOC_response(model_trained, check_xset,Y1_scaler,device):
    feature = 16
    ranges = [-0.5,5]
    interval = 0.25
    Tx=365
    tyear = 18
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Rh_daily = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Rh_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y1_pred_t,___, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Rh_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,1]
        
    for i in range(total_n):
        Rh_daily[i] = torch.mean(Z_norm_reverse(Rh_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[1,:],1.0))
        if i > 0:
            slopes[i-1] = (Rh_daily[i]-Rh_daily[i-1])/interval
    loss = torch.sum(ReLU1(slopes))   
    return loss 

def check_Rh2GPP_response(model_trained, check_xset,Y1_scaler,device):
    feature = 8
    ranges = [-0.5,5]
    interval = 0.25
    Tx=365
    tyear = 18
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Rh_daily = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Rh_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y1_pred_t,___, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Rh_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,1]
        
    for i in range(total_n):
        Rh_daily[i] = torch.mean(Z_norm_reverse(Rh_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[1,:],1.0))
        if i > 0:
            slopes[i-1] = (Rh_daily[i]-Rh_daily[i-1])/interval
    loss = torch.sum(ReLU1(slopes))
    return loss 

class SPLoss_Yield_v2(nn.Module):
    def __init__(self,threshold, growing_factor):
        super(SPLoss_Yield_v2, self).__init__()
        self.threshold = threshold
        self.growing_factor = growing_factor
        
    def forward(self, corn_pred, corn_obs, soybean_pred, soybean_obs, mask_corn, mask_soybean, val_flag,device):
        #Yield loss, 2 dim is corn or soybean
        super_loss_corn = torch.masked_select((corn_pred - corn_obs)**2,mask_corn.ge(0.5))
        if len(super_loss_corn) ==0:
            super_loss_corn = torch.tensor(0.0,device=device)
        super_loss_soybean = torch.masked_select((soybean_pred - soybean_obs)**2,mask_soybean.ge(0.5))
        if len(super_loss_soybean) ==0:
            super_loss_soybean = torch.tensor(0.0,device=device)
        if val_flag:
            loss = (torch.mean(super_loss_corn) + torch.mean(super_loss_soybean))/2.0
        else:
            v_corn,v_soybean = self.spl_loss(super_loss_corn,super_loss_soybean)
            loss = (torch.mean(super_loss_corn*v_corn) + torch.mean(super_loss_soybean*v_soybean))/2.0
        return loss
        
    def increase_threshold(self):
        self.threshold *= self.growing_factor
        
    def spl_loss(self, super_loss1, super_loss2):
        v1 = super_loss1 < self.threshold
        v2 = super_loss2 < self.threshold
        return v1.float(),v2.float()
       
ReLU1 = nn.ReLU()    
def check_yield_response_loss(model_trained, check_xset, feature, ranges, interval,device): 
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Y2_annual = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    Tx=365
    tyear = 18
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Y2_pred=torch.zeros([check_n*total_n,tyear],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        __,Y2_pred_t, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Y2_pred[:,it] = Y2_pred_t[:,0,0]
        
    for i in range(total_n):
        Y2_annual[i] = torch.mean(Y2_pred[i*check_n:(i+1)*check_n,:])
        if i > 0:
            slopes[i-1] = (Y2_annual[i]-Y2_annual[i-1])/interval
    loss = torch.sum(ReLU1(-slopes))   
    return loss

def check_yield_response_loss_v2(model_trained, check_xset, feature, ranges, interval,device): 
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Y2_annual = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    Tx=365
    tyear = 18
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Y2_pred=torch.zeros([check_n*total_n,tyear],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        ___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y2_pred_t, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Y2_pred[:,it] = Y2_pred_t[:,0,0]
        
    for i in range(total_n):
        Y2_annual[i] = torch.mean(Y2_pred[i*check_n:(i+1)*check_n,:])
        if i > 0:
            slopes[i-1] = (Y2_annual[i]-Y2_annual[i-1])/interval
    loss = torch.sum(ReLU1(-slopes))   
    return loss

def Check_yield_GPP(Yield, GPP):
    #0<Yield < 0.5*GPP, absolute values
    loss = torch.mean(ReLU1(Yield - 0.5*GPP))+torch.mean(ReLU1(0.0 - Yield))
    return loss

def myloss_mb_flux_mask_re_v2(Ra_pred, Rh_pred, NEE_pred, Reco_obs, NEE_obs, X_sites,GPP_scaler, \
                              Y1_scaler,Y1_mask, lamda,tol_MB):
   
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    Tx = 365
    Ra_tol = 0.001
    for bb in range(len(Reco_obs)):
        #Ra, Rh and NEE loss by set DOY < 105 and DOY > 300 are Rh
        for y in range(int(Reco_obs[bb].size(0)/Tx)):
            loss1 = loss1 + torch.mean(Y1_mask[bb][y*Tx:y*Tx+105].view(-1)*(Rh_pred[bb][y*Tx:y*Tx+105].view(-1) - \
                                                                        Reco_obs[bb][y*Tx:y*Tx+105].view(-1))**2) + \
                    torch.mean(Y1_mask[bb][y*Tx+300:(y+1)*Tx].view(-1)*(Rh_pred[bb][y*Tx+300:(y+1)*Tx].view(-1) - \
                                                                        Reco_obs[bb][y*Tx+300:(y+1)*Tx].view(-1))**2) +\
                    torch.mean(Y1_mask[bb][y*Tx+105:y*Tx+300].view(-1)*(Ra_pred[bb][y*Tx+105:y*Tx+300].view(-1) + \
                                    Rh_pred[bb][y*Tx+105:y*Tx+300].view(-1)-Reco_obs[bb][y*Tx+105:y*Tx+300].view(-1))**2)
        loss1 = loss1 + torch.mean(Y1_mask[bb].view(-1)*(NEE_pred[bb].view(-1) - NEE_obs[bb].view(-1))**2)
        #mass balance loss GRR+Ra+Rh-NEE = 0   (ra, rh are negative)
        GPP = Z_norm_reverse(X_sites[bb][:,:,8],GPP_scaler,1.0).view(-1)
        Reco = Ra_pred[bb].view(-1) + Rh_pred[bb].view(-1)
        NEE = NEE_pred[bb].view(-1)
        loss2 =  loss2 + torch.mean(Y1_mask[bb]*(ReLU1(torch.abs(GPP - Reco + NEE) - tol_MB*torch.abs(Reco))))
        ####bias**2
        loss3 = loss3 + torch.abs(torch.mean(Y1_mask[bb].view(-1)*(Ra_pred[bb].view(-1) + Rh_pred[bb].view(-1) - \
                                                                   Reco_obs[bb].view(-1))))
    loss = lamda[0]*loss1/float(len(Reco_obs)) + lamda[1]*loss2/float(len(Reco_obs))
    return loss,loss1,loss2,loss3

def myloss_mb_flux_mask_re(Reco_pred, NEE_pred, Reco_obs, NEE_obs, X_sites, GPP_scaler, Y1_scaler,Y1_mask, lamda,tol_MB):
   
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    for bb in range(len(Reco_pred)):
         #Ra, Rh and NEE loss
        loss1 = loss1 + torch.mean(Y1_mask[bb].view(-1)*(Reco_pred[bb].view(-1) - Reco_obs[bb].view(-1))**2) +\
                torch.mean(Y1_mask[bb].view(-1)*(NEE_pred[bb].view(-1) - NEE_obs[bb].view(-1))**2)
        #mass balance loss GRR+Ra+Rh-NEE = 0   (ra, rh are negative)
        GPP = Z_norm_reverse(X_sites[bb][:,:,8],GPP_scaler,1.0).view(-1)
        Reco = Reco_pred[bb].view(-1)
        NEE = NEE_pred[bb].view(-1)
        loss2 =  loss2 + torch.mean(Y1_mask[bb]*(ReLU1(torch.abs(GPP - Reco + NEE) - tol_MB*torch.abs(Reco))))
        ####bias**2
        loss3 = loss3 + torch.abs(torch.mean(Y1_mask[bb].view(-1)*(Reco_pred[bb].view(-1) - Reco_obs[bb].view(-1))))
    loss = lamda[0]*loss1/float(len(Reco_pred)) + lamda[1]*loss2/float(len(Reco_pred))
    return loss,loss1,loss2,loss3

def check_Ra_Rh_response_v2(model_trained, check_xset, feature, ranges, interval,Ra_r_min, Ra_org,Rh_r_min, Rh_org,Y1_scaler, device):
    Tx=365
    tyear = 18
    compute_r2 = R2Loss()
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Ra_daily = torch.zeros([total_n],device=device)
    Rh_daily = torch.zeros([total_n],device=device)
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Ra_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    Rh_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y1_pred_t,___, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Ra_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,0]
        Rh_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,1]
        
    for i in range(total_n):
        Ra_daily[i] = torch.mean(Z_norm_reverse(Ra_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[0,:],1.0))
        Rh_daily[i] = torch.mean(Z_norm_reverse(Rh_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[1,:],1.0)) 
    if feature == 16:
        loss = ReLU1(Ra_r_min - pearsonr(Ra_daily,Ra_org)) + ReLU1(0.99 - compute_r2(Rh_daily,Rh_org))
    else:
        loss = ReLU1(Ra_r_min - pearsonr(Ra_daily,Ra_org)) + ReLU1(Rh_r_min - pearsonr(Rh_daily,Rh_org))
    return loss

ReLU1 = nn.ReLU() 
def Get_Ra_Rh_org(model_trained, check_xset,feature, ranges, interval,Y1_scaler, device):
    Tx=365
    tyear = 18
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Ra_daily = torch.zeros([total_n],device=device)
    Rh_daily = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Ra_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    Rh_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y1_pred_t,___, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Ra_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,0]
        Rh_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,1]
        
    for i in range(total_n):
        Ra_daily[i] = torch.mean(Z_norm_reverse(Ra_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[0,:],1.0))
        Rh_daily[i] = torch.mean(Z_norm_reverse(Rh_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[1,:],1.0))
    return Ra_daily, Rh_daily 

def check_Rh_response_v2(model_trained, check_xset, feature, ranges, interval,r_min, Rh_org,Y1_scaler, device):
    Tx=365
    tyear = 18
    total_n = int((ranges[1] - ranges[0])/interval)+1
    Rh_daily = torch.zeros([total_n],device=device)
    slopes = torch.zeros([total_n-1],device=device)
    check_n = check_xset.size(0)
    bsz = check_n
    perbutated_check_xset = torch.zeros([check_n*total_n,check_xset.size(1),check_xset.size(2)],device=device)
    for i in range(total_n):
        replace_v = ranges[0] + interval*i
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,:] = check_xset
        perbutated_check_xset[i*check_n:(i+1)*check_n,:,feature] = replace_v
    Rh_pred=torch.zeros([check_n*total_n,tyear*Tx],device=device)
    hidden = model_trained.init_hidden(check_n*total_n)
    ##spinup 2 years for turnover
    for it in range(2):
        __,___,hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
    for it in range(tyear):
        Y1_pred_t,___, hidden = model_trained(perbutated_check_xset[:,Tx*it:Tx*it+Tx,:],hidden)
        Rh_pred[:,Tx*it:Tx*it+Tx] = Y1_pred_t[:,:,1]
        
    for i in range(total_n):
        Rh_daily[i] = torch.mean(Z_norm_reverse(Rh_pred[i*check_n:(i+1)*check_n,:],Y1_scaler[1,:],1.0))
    r = pearsonr(Rh_daily,Rh_org)
    loss = ReLU1(r_min - r)
    return loss

#for multi-task learning, sumloss
def sum_flux_mask_re(Reco_pred, NEE_pred, Reco_obs, NEE_obs,Y1_mask):
    loss1 = 0.0
    for bb in range(len(Reco_pred)):
         #Ra, Rh and NEE loss
        loss1 = loss1 + torch.mean(Y1_mask[bb].view(-1)*(Reco_pred[bb].view(-1) - Reco_obs[bb].view(-1))**2) +\
                torch.mean(Y1_mask[bb].view(-1)*(NEE_pred[bb].view(-1) - NEE_obs[bb].view(-1))**2)
    loss = loss1/float(len(Reco_pred)) 
    return loss

##########################################################################################
#loss function customization
#
##########################################################################################
import ast
from collections import OrderedDict
import re

# Must define this function here, otherwise the namespace = globals() can't find this function

def safe_repr(value):
    """
    Returns a string representation of 'value'.
    - For basic Python types (int, float, str, list, tuple, set), uses repr() directly.
    - For NumPy arrays, Pandas DataFrames, and Torch Tensors,
      converts them to a Python list before using repr().
    """
    if isinstance(value, (int, float, str, list, tuple, set, dict, bool, type(None))):
        return repr(value)
    elif isinstance(value, np.ndarray):
        return repr(value.tolist())
    elif isinstance(value, pd.DataFrame):
        # Convert DataFrame to a list of lists (rows) or dict of lists (columns)
        # Choosing rows here for a common list-like representation
        return repr(value.values.tolist())
    elif isinstance(value, torch.Tensor):
        return repr(value.tolist())
    else:
        # Fallback for other types
        return repr(value)

class LossFunctionCompiler:
    def __init__(self, script_config):
        self.script_config = script_config
        self.validate_config()
        self.analyze_dependencies()
        self.generate_class_code()
        
    def validate_config(self):
        """Validate configuration format"""
        required_sections = ['parameters', 'variables', 'loss_formula']
        for section in required_sections:
            if section not in self.script_config:
                raise ValueError(f"Script config must contain '{section}' section")
                
        # Ensure the loss formula contains the final loss expression
        if 'loss' not in self.script_config['loss_formula']:
            raise ValueError("Loss formula must contain a 'loss' expression")
    
    def extract_all_expressions(self):
        """Extracts all expressions into a dictionary"""
        expressions = {}
        
        # Add parameters
        expressions.update(self.script_config['parameters'])
        
        # Add variable expressions
        expressions.update(self.script_config['variables'])
        
        # Add loss formulas
        expressions.update(self.script_config['loss_formula'])
        
        return expressions
    
    def analyze_dependencies(self):
        """Analyzes dependencies between expressions"""
        # Extract all expressions
        all_expressions = self.extract_all_expressions()
        
        self.dependencies = OrderedDict()
        self.tensor_extractions = []
        self.intermediate_exprs = []
        
        # Define list of known function names (these should not be considered variable dependencies)
        self.known_functions = {'mean', 'abs', 'sum', 'exp', 'log', 'sqrt', 'min', 'max', 'Z_norm_reverse', 'relu'}
        
        # 1. Create dependency graph
        for key, expr in all_expressions.items():
            if not isinstance(expr, str):
                continue
                
            try:
                tree = ast.parse(expr, mode='eval')
            except SyntaxError as e:
                raise ValueError(f"Invalid expression for '{key}': {e}")
                
            variables = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id != 'self' and node.id not in self.known_functions:
                    variables.add(node.id)
            
            self.dependencies[key] = variables
        
        # 2. Identify tensor extraction expressions (those that depend only on inputs or class attributes)
        for key, deps in self.dependencies.items():
            if key == 'loss':
                continue  # Skip final loss
                
            # Check if it only depends on input tensors or configuration parameters
            if all(dep in ['batch_x', 'y_pred', 'y_true'] or 
                   (dep in all_expressions and not isinstance(all_expressions[dep], str))
                   for dep in deps):
                self.tensor_extractions.append(key)
        
        # 3. Identify intermediate expressions (dependent on other expressions)
        self.intermediate_exprs = [key for key in self.dependencies.keys() 
                                  if key != 'loss' and key not in self.tensor_extractions]
        
        # 4. Determine evaluation order for intermediate expressions
        evaluated = set(self.tensor_extractions)
        available_vars = set(self.tensor_extractions) | set(
            k for k, v in all_expressions.items() if not isinstance(v, str)
        )
        
        # Topological sort
        evaluation_order = []
        while self.intermediate_exprs:
            added = False
            remaining = []
            
            for key in self.intermediate_exprs:
                deps = self.dependencies[key]
                
                # Check if all dependencies are available
                if deps.issubset(available_vars):
                    evaluation_order.append(key)
                    available_vars.add(key)
                    added = True
                else:
                    remaining.append(key)
            
            if not added and remaining:
                # Find missing dependencies
                missing_deps = {}
                for key in remaining:
                    deps = self.dependencies[key]
                    missing = deps - available_vars
                    if missing:
                        missing_deps[key] = missing
                
                error_msg = "Circular dependency detected or missing variables:\n"
                for key, missing in missing_deps.items():
                    error_msg += f"  - '{key}' missing: {', '.join(missing)}\n"
                raise RuntimeError(error_msg)
            
            self.intermediate_exprs = remaining
        
        self.evaluation_order = evaluation_order
    
    def replace_shortened_functions(self, expr):
        """Replace abbreviated function names with their full torch-prefixed forms"""
        replacements = {
            r'\bmean\(': 'torch.mean(',
            r'\babs\(': 'torch.abs(',
            r'\bsum\(': 'torch.sum(',
            r'\bexp\(': 'torch.exp(',
            r'\blog\(': 'torch.log(',
            r'\bsqrt\(': 'torch.sqrt(',
            r'\bmin\(': 'torch.min(',
            r'\bmax\(': 'torch.max(',
            r'\brelu\(': 'torch.relu(',
            # r'\bZ_norm_reverse\(': 'self.Z_norm_reverse(',
        }
        
        for pattern, replacement in replacements.items():
            expr = re.sub(pattern, replacement, expr)
        
        return expr
    
    def generate_class_code(self):
        """Generate Python source code for the CarbonFluxLoss class"""
        # Get parameter configuration
        parameters = self.script_config['parameters']
        all_expressions = self.extract_all_expressions()
        
        # Class header
        class_code = [
            "import torch",
            "import torch.nn as nn",
            "",
            "class CarbonFluxLoss(nn.Module):",
        ]
        
        # Generate __init__ method parameter list
        init_params = []
        for key, value in parameters.items():
            # For scaler parameters, use default value None
            # if key in ['GPP_scaler', 'y_scaler']:
            #     init_params.append(f"{key}=None")
            # else:
            #     init_params.append(f"{key}={repr(value)}")

            init_params.append(f"{key}={safe_repr(value)}")
                
        class_code.append(f"    def __init__(self, {', '.join(init_params)}):")
        class_code.append("        super().__init__()")
        
        # Store all parameters as class attributes
        for key in parameters.keys():
            # For scalers, store them as tensors that do not require gradients
            # if key in ['GPP_scaler', 'y_scaler']:
            #     class_code.append(f"        self.{key} = torch.tensor({key}, requires_grad=False) if {key} is not None else None")
            # else:
            #     class_code.append(f"        self.{key} = {key}")

            class_code.append(f"        self.{key} = {key}")
        
        # Add Z_norm_reverse method
        # class_code.extend([
        #     "",
        #     "    def Z_norm_reverse(self, x, scaler):",
        #     "        \"\"\"Z norm reverse\"\"\"",
        #     "        if scaler is None:",
        #     "            return x",
        #     "        return x * scaler[1] + scaler[0]"
        # ])
        
        # Process other non-parameter attributes (if any)
        for key, value in all_expressions.items():
            if key in parameters:  # Skip processed parameters
                continue
                
            if not isinstance(value, str):
                if isinstance(value, (int, float)):
                    class_code.append(f"        self.{key} = {value}")
                # If the parameter is a tensor, store it as a tensor that does not require gradients
                elif torch.is_tensor(value):
                    class_code.append(f"        self.{key} = torch.tensor({value.tolist()}, requires_grad=False)")
                # Other types are stored directly
                else:
                    class_code.append(f"        self.{key} = {repr(value)}")
        
        # Forward method header
        class_code.extend([
            "",
            "    def forward(self, y_pred, y_true, batch_x):"
        ])
        
        # Copy parameters to local variables
        param_names = list(parameters.keys())
        if param_names:
            class_code.append("        # Copy parameters to local variables")
            for name in param_names:
                class_code.append(f"        {name} = self.{name}")
        
        # Tensor extraction
        if self.tensor_extractions:
            class_code.append("\n        # Tensor extraction")
            for key in self.tensor_extractions:
                expr = all_expressions[key]
                # Replace shorthand function names with full torch-prefixed forms
                expr = self.replace_shortened_functions(expr)
                class_code.append(f"        {key} = {expr}")
        
        # Intermediate calculations
        if self.evaluation_order:
            class_code.append("\n        # Intermediate calculations")
            for key in self.evaluation_order:
                expr = all_expressions[key]
                # Replace shorthand function names with full torch-prefixed forms
                expr = self.replace_shortened_functions(expr)
                class_code.append(f"        {key} = {expr}")
        
        # Add loss
        class_code.append("\n        # Loss")
        loss_expr = self.replace_shortened_functions(all_expressions['loss'])
        class_code.append(f"        loss = {loss_expr}")
        class_code.append("        return loss")
        
        self.class_code = "\n".join(class_code)
    
    def compile_class(self):
        """Compile and return the CarbonFluxLoss class"""
        # namespace = {
        #     'torch': torch,
        #     'nn': nn,
        #     '__name__': '__carbon_flux_loss__'
        # }
        namespace = globals()
        
        try:
            exec(self.class_code, namespace)
        except Exception as e:
            print("Generated code:")
            print(self.class_code)
            raise RuntimeError(f"Failed to compile CarbonFluxLoss class: {e}")
        
        return namespace['CarbonFluxLoss']
    
    def generate_class(self):
        return self.compile_class()



##########################################################################################
#model architecture design
#
##########################################################################################
import re
from time_series_models import TimeSeriesModel, Attention

def extract_functions(expr: str) -> tuple[list[str], list[str]]:
    """
    Given an expression like 'fc(dropout(rh_out))' or 'attend & x',
    return a list of function names and their direct arguments.
    - For 'fc(dropout(rh_out))' returns ['fc', 'dropout'], ['rh_out']
    - For 'nn.Linear(64,32)' returns ['nn.Linear'], ['64', '32']
    - For 'attend & x' (no function) returns ['attend & x']
    """
    # 1) Find all function names (identifier followed by '(')
    func_names = re.findall(r'\b(\w+(?:\.\w+)*)\s*\(', expr)

    # 2) Capture the innermost parenthesis content (no nested '(' or ')')
    inner_args = re.findall(r'\(\s*([^()]+?)\s*\)', expr)
    # Filter out any that still contain parentheses
    args = [arg for arg in inner_args if '(' not in arg and ')' not in arg]
    # args will like ['64, 32']
    parts = [p.strip() for p in args[0].split(',')] if args else []

    # 3) If we found any functions, return them plus the direct args
    if func_names:
        return func_names, parts

    # 4) Otherwise there’s no function call—return the raw expression
    return [], [expr.strip()]


class ModelStructureCompiler:
    """
    Compiles a PyTorch TimeSeriesModel subclass based on a configuration dict.

    Config schema:
      class_name: str
      base_class: nn.Module subclass
      init_params: dict of constructor parameters and their default values
      layers: dict mapping attribute names to (factory_fn_name: str, *args)
              where factory_fn_name can be 'gru', 'lstm', 'linear', 'dropout'
      forward: dict mapping var names to Python expressions (strings)

    Generates a class that:
      - Inherits from base_class
      - Defines __init__ with parameters and defaults
      - Calls super().__init__(<init_param_names>)
      - Instantiates layers with auto kwargs for recurrent layers
      - Defines forward(self, x) based on forward config, translating '+' to torch.cat and layer calls to [0] captures
    """
    def __init__(self, config: dict):
        self.cfg = config
        self._validate()
        # store layer keys for use in replacement
        self.layer_names = list(self.cfg['layers'].keys())

    def _validate(self):
        required = ['class_name', 'base_class', 'init_params', 'layers', 'forward']
        for key in required:
            if key not in self.cfg:
                raise ValueError(f"Missing config key: {key}")
        if not isinstance(self.cfg['init_params'], dict):
            raise ValueError("'init_params' must be a dict of name->default values")
        if not isinstance(self.cfg['layers'], dict) or not isinstance(self.cfg['forward'], dict):
            raise ValueError("'layers' and 'forward' must be dicts")

    def replace_shortened_functions(self, expr: str) -> str:
        """
        Replace shorthand layer calls with self-prefixed calls
        based on configured layer names.
        """
        for name in self.layer_names:
            # replace e.g. 'fc(' with 'self.fc('
            expr = re.sub(rf"\b{name}\(", f"self.{name}(", expr)
        return expr

    def check_configuration(self):
        init_params = self.cfg['init_params']
        layers_cfg = self.cfg['layers']
        forward_cfg = self.cfg['forward']
        
        # Instantiate layers
        layers_dict = {}
        layers_name = list()
        fn_list = list() # record all functions in layers
        layers_keys = list(layers_cfg.keys())
        for attr, spec in layers_cfg.items():
            fn = spec[0].strip().lower()
            fn_list.append(fn)
            layers_name.append(attr.strip())

            args = spec[1:]
            if fn in ('gru', 'lstm'):
                inp, hid, *rest = args
                cls = 'nn.GRU' if fn == 'gru' else 'nn.LSTM'
                outp = hid
                layers_dict[attr] = [cls, inp, outp]
            elif fn == 'linear':
                inp, outp = args
                layers_dict[attr] = ['nn.Linear', inp, outp]
            elif fn == 'dropout':
                p, = args
                layers_dict[attr] = ['nn.Dropout', 0, 0]
            elif fn == 'attention':
                p, outp = args
                layers_dict[attr] = ['Attention', p, outp]
            # Activation: ReLU
            elif fn == 'relu':
                layers_dict[attr] = ['nn.ReLU', 0, 0]
            # Activation: Tanh
            elif fn == 'tanh':
                layers_dict[attr] = ['nn.Tanh', 0, 0]
            elif fn == 'softmax' or fn == 'F.softmax':
                layers_dict[attr] = ['F.softmax', 0, 0]
            # Sequential container
            elif fn in ('sequential', 'nn.sequential'):
                for func_call in args:
                    fn,params = extract_functions(func_call)
                    if fn == ['nn.Linear']:
                        inp, outp = params
                layers_dict[attr] = ['nn.Sequential', 0, outp] # get last nn.Linear()
            else:
                inp, outp = args
                layers_dict[attr] = [fn, inp, outp]

        fn_set = set(fn_list) #remove duplicate items

        # Forward process
        concat_symbol = '&'
        mm_symbol = '@'
        dot_symbol = '.'

        return_p_list = ['x']
        output_p_dict = dict()
        init_params_keys = list(init_params.keys())
        output_p_dict['x'] = init_params_keys[0] # Get first key

        for var, expr in forward_cfg.items():
            # get return parameters
            return_p = [part.strip() for part in var.split(',')]
            return_p_list += return_p

            if concat_symbol in expr and 'torch.' not in expr and '(' not in expr:
                parts = [p.strip() for p in expr.split(concat_symbol)]
                # each part should exist in previous
                for part in parts:
                    if part not in return_p_list:
                        print(f"Warning check {part} in the {expr}")

                total_dim = []
                for part in parts:
                    _dim = output_p_dict[part]
                    total_dim.append(_dim)

                full_dim = '+'.join(total_dim)
                for p in return_p:
                    output_p_dict[p] = full_dim

            elif mm_symbol in expr :
                parts = [p.strip() for p in expr.split(mm_symbol)]
                # Get last matrix's dimension
                last_part = parts[-1]
                for p in return_p:
                    if last_part in return_p_list:
                        output_p_dict[p] = output_p_dict[last_part]
                    else:
                        output_p_dict[p] = -1 # means unknow dimension

            elif dot_symbol in expr: # exist like 0.5 F.softmax(), torch.sqrt(), nn.zeros()
                skip_line = False
                fns,params = extract_functions(expr) # get [fns], [params]
                if len(fns) == 0: # no function exist
                    skip_line = True
                else:
                    for fn in fns:
                        if dot_symbol in fn:
                            skip_line = True
                
                # do nothing, skip this line
                if skip_line:
                    continue

            else:
                # extract func names and parameters
                fns,params = extract_functions(expr) # get [fns], [params]
                fns.reverse() # change func order for case: ['fc', 'dropout'], change to ['dropout', 'fc']
                if len(fns) == 0: # no function in configuration the the fns is a empty list
                    output_p_dict[var] = expr
                    continue

                # Input parameter should exist in before rows
                for param in params: # Only support one param now
                    if param not in return_p_list: # The param should be pre lines output
                        print(f"Warning check {param} in the {expr}")
                
                # the function name should exist in layers
                for fn in fns: 
                    if fn not in layers_keys:
                        print(f"Invalid function name {fn} in the {expr}")

                # get each func call returned dimension
                for fn in fns:
                    input_dim, output_dim = layers_dict[fn][1:]
                    if output_dim == 0: 
                        # don't change dimension, need get original dim from input_dim
                        output_dim = output_p_dict[params[0]]

                    for p in return_p:
                        output_p_dict[p] = output_dim

                for param in params:
                    if param == 'x':
                        continue

                    if input_dim == 0:
                        continue

                    try:
                        input_para_dim_value = eval(output_p_dict[param], {}, init_params)
                        layer_required_dim = eval(input_dim, {}, init_params)
                    except NameError as e:
                        print(f"Caught a ValueError: {e}")
                        print(f"Warning: check {param} in {expr}")

                    if input_para_dim_value != layer_required_dim:
                        print(f"Warning: check {param} in {expr}. The dim of {param} is {input_para_dim_value}, but layer {fn} requires a input_dim of {layer_required_dim}")


    def generate_model(self):
        class_name = self.cfg['class_name']
        base = self.cfg['base_class']
        init_params = self.cfg['init_params']
        layers_cfg = self.cfg['layers']
        forward_cfg = self.cfg['forward']

        lines = []
        # Class header
        lines.append(f"class {class_name}({base}):")
        # __init__ signature
        params_sig = ', '.join(f"{k}={repr(v)}" for k, v in init_params.items())
        lines.append(f"    def __init__(self, {params_sig}):")
        # super init
        if base == 'TimeSeriesModel':
            names = ', '.join(init_params.keys())
            lines.append(f"        super().__init__({names})")
        else:
            lines.append(f"        super().__init__()")
        lines.append("")

        # Save input parameters
        for key in init_params.keys():
            lines.append(f"        self.{key} = {key}")
        lines.append("")

        # Instantiate layers
        for attr, spec in layers_cfg.items():
            fn = spec[0].lower()
            args = spec[1:]
            if fn in ('gru', 'lstm'):
                inp, hid, nl, dp, *rest = args
                cls = 'nn.GRU' if fn == 'gru' else 'nn.LSTM'
                lines.append(
                    f"        self.{attr} = {cls}("
                    f"{inp}, {hid}, {nl}, bias=True, batch_first=True, dropout={dp})"
                )
            elif fn == 'linear':
                inp, outp = args
                lines.append(f"        self.{attr} = nn.Linear({inp}, {outp})")
            elif fn == 'dropout':
                p, = args
                lines.append(f"        self.{attr} = nn.Dropout({p})")
            elif fn == 'attention':
                p, outp = args
                lines.append(f"        self.{attr} = Attention({p})")
            # Activation: ReLU
            elif fn == 'relu':
                lines.append(f"        self.{attr} = nn.ReLU()")
            # Activation: Tanh
            elif fn == 'tanh':
                lines.append(f"        self.{attr} = nn.Tanh()")
            # Sequential container
            elif fn in ('sequential', 'nn.sequential'):
                # args are module definition strings
                lines.append(f"        self.{attr} = nn.Sequential(")
                for module_str in args:
                    lines.append(f"            {module_str},")
                # remove trailing comma from last
                lines[-1] = lines[-1].rstrip(',')
                lines.append(f"        )")
            else:
                args_list = ', '.join(map(str, args[:-1])) # the last is the output dimmension
                lines.append(f"        self.{attr} = {fn}({args_list})")

        # forward method
        lines.append("")
        lines.append("    def forward(self, x: torch.Tensor):")
        # Translate forward config to code lines

        # copy parameters to local var
        lines.append("        # Copy parameter to local")
        for key in init_params.keys():
            lines.append(f"        {key} = self.{key}")
        lines.append("")

        concat_symbol = '&'
        mm_symbol = '@'
        for var, expr in forward_cfg.items():
            # handle concatenation
            if concat_symbol in expr and 'torch.' not in expr and '(' not in expr:
                parts = [p.strip() for p in expr.split(concat_symbol)]
                concat = ', '.join(parts)
                code = f"        {var} = torch.cat([{concat}], dim=-1)"
            # handle layer calls e.g. 'fc(x), dropout(x), fc(dropout(x))'
            elif mm_symbol in expr :
                parts = [p.strip() for p in expr.split(mm_symbol)]
                matrix_multiple = ', '.join(parts)
                code = f"        {var} = torch.bmm({matrix_multiple})"
            else:
                # apply shorthand replacement
                expr = self.replace_shortened_functions(expr)
                code = f"        {var} = {expr}"
            lines.append(code)
        lines.append("        return output")

        # Execute dynamic code\        
        code_str = '\n'.join(lines)
        self.class_code = code_str
        # exec_globals = {
        #     're': re,
        #     'torch': torch,
        #     'nn': nn,
        #     'F': F,
        #     base.__name__: base,
        #     'Attention': Attention
        # }
        # namespace = {}
        # exec(code_str, exec_globals, namespace)

        namespace = globals()
        exec(code_str, namespace)
        return namespace[class_name]

