##############################################################################
## Copyright (C) 2023 - All Rights Reserved
## This file is part of the manuscript named "Knowledge-based artificial 
## intelligence significantly improved agroecosystem carbon cycle 
## quantification". Unauthorized copying/distributing/modifying of this file, 
## via any medium is strictly prohibited.
## Proprietary and confidential Written by DBRP authors of above manuscript
#############################################################################
import numpy as np
import math
import os
from io import open
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess as sp
import os
import copy

##########################################################################################
#basic functions
#
##########################################################################################
class R2Loss(nn.Module):
    #calculate coefficient of determination
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y
def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

def Z_norm(X):
    X_mean=X.numpy().mean(dtype=np.float64)
    X_std=np.std(np.array(X,dtype=np.float64))
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

##########################################################################################
#Models
#
##########################################################################################

# use Reco model v1--GRU model
class RecoGRU_multitask_v8_1(nn.Module):
    def __init__(self, ninp, dropout, seq_len, GPP_scaler, Ra_scaler, Yield_scaler,Res_scaler,GPP_Res_fmean):
        super(RecoGRU_multitask_v8_1, self).__init__()
        nhid = 64
        nlayers = 2
        self.nhid = nhid
        self.nlayers = nlayers
        self.GPP_scaler = GPP_scaler
        self.Ra_scaler = Ra_scaler
        self.Yield_scaler = Yield_scaler
        self.Res_scaler = Res_scaler
        self.GPP_Res_fmean = GPP_Res_fmean
        self.seq_len = seq_len
        self.gru_basic = nn.GRU(ninp, nhid,nlayers,dropout=dropout, batch_first=True)
        self.gru_Ra = nn.GRU(nhid+ninp, nhid,1,batch_first=True)
        self.gru_RhNEE = nn.GRU(nhid+ninp+1, nhid,nlayers,dropout=dropout, batch_first=True)#+1 means res ini 
        self.drop=nn.Dropout(dropout)
        self.densor_Ra = nn.Linear(nhid, 1)
        self.densor_RhNEE = nn.Linear(nhid, 2)
        #attn for yield prediction
        self.attn = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yield = nn.Sequential(
            nn.Linear(nhid, 64),
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
        self.densor_RhNEE.bias.data.zero_()
        self.densor_RhNEE.weight.data.uniform_(-initrange, initrange)
        for ii in range(4):
            self.attn[ii*2].bias.data.zero_()
            self.attn[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yield[ii*2].bias.data.zero_()
            self.densor_yield[ii*2].weight.data.uniform_(-initrange, initrange)
        

    def forward(self, inputs, hidden):
        output, hidden1 = self.gru_basic(inputs, hidden[1])
        #predict yield
        inputs2 = self.drop(output)
        attn_weights = F.softmax(self.attn(inputs2), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs2 = torch.bmm(attn_weights,inputs2)
        output2 = self.densor_yield(inputs2)
        
        #predict flux
        #Ra
        output1 , hidden2 = self.gru_Ra(torch.cat((self.drop(output),inputs), 2), hidden[2])
        Ra = self.densor_Ra(self.drop(output1))
        
        #RhNEE
        Res_ini = hidden[0]
        output1, hidden3  = self.gru_RhNEE(torch.cat((Res_ini.repeat(1,self.seq_len,1),\
                                                      self.drop(output),inputs), 2), hidden[3])
        output1 = torch.cat((Ra,self.densor_RhNEE(self.drop(output1))),2)

        #caculate annual Res
        #Annual GPP+ Annual Ra - Yield, GPP is no.8 inputs
        annual_GPP = torch.sum(Z_norm_reverse(inputs[:,:,8],self.GPP_scaler),dim=1).view(-1,1,1)
        annual_Ra = torch.sum(Z_norm_reverse(Ra[:,:,0],self.Ra_scaler),dim=1).view(-1,1,1)
        annual_Yield = Z_norm_reverse(output2[:,0,0],self.Yield_scaler).view(-1,1,1)
        #control 0< Res_ini < GPP
        Res_ini = self.ReLU(annual_GPP+annual_Ra - annual_Yield)
        Res_ini[Res_ini > annual_GPP] = annual_GPP[Res_ini > annual_GPP]
        #scale Res_ini
        Res_ini = Z_norm_with_scaler(Res_ini,self.Res_scaler)
        
        return output1, output2, (Res_ini,hidden1,hidden2,hidden3)
#bsz should be batch size
    def init_hidden(self, bsz,GPP_total):
        Res_ini = (torch.sum(Z_norm_reverse(GPP_total,self.GPP_scaler,1.0),dim=1)/ \
                  (float(GPP_total.size(1))/float(self.seq_len))/self.GPP_Res_fmean).view(-1,1,1)
        Res_ini = Z_norm_with_scaler(Res_ini,self.Res_scaler)
        weight = next(self.parameters())
        return (Res_ini,
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

# use Reco model v1--GRU model
class RecoGRU_multitask_v11_3(nn.Module):
    def __init__(self, ninp, dropout, seq_len, GPP_scaler, Ra_scaler, Yield_scaler,Res_scaler):
        super(RecoGRU_multitask_v11_3, self).__init__()
        nhid = 64
        self.nhid = nhid
        self.GPP_scaler = GPP_scaler
        self.Ra_scaler = Ra_scaler
        self.Yield_scaler = Yield_scaler
        self.Res_scaler = Res_scaler
 #       self.GPP_Res_fmean = GPP_Res_fmean
        self.seq_len = seq_len
        self.gru_basic = nn.GRU(ninp,nhid,2,dropout=dropout, batch_first=True)
        self.gru_Ra = nn.GRU(nhid+ninp, nhid,1,batch_first=True)
        self.gru_Rh = nn.GRU(nhid+ninp+1, nhid,2,dropout=dropout, batch_first=True)#+1 means res ini 
        self.gru_NEE = nn.GRU(ninp+2, nhid,1, batch_first=True)#+2 Ra and Rh
        self.drop=nn.Dropout(dropout)
        self.densor_Ra = nn.Linear(nhid, 1)
        self.densor_Rh = nn.Linear(nhid, 1)
        self.densor_NEE = nn.Linear(nhid, 1)
        #attn for yield prediction
        self.attn = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yield = nn.Sequential(
            nn.Linear(nhid, 64),
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
        

    def forward(self, inputs, hidden):
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
        annual_GPP = torch.sum(Z_norm_reverse(inputs[:,:,8],self.GPP_scaler),dim=1).view(-1,1,1)
        annual_Ra = torch.sum(Z_norm_reverse(Ra[:,:,0],self.Ra_scaler),dim=1).view(-1,1,1)
        annual_Yield = Z_norm_reverse(output2[:,0,0],self.Yield_scaler).view(-1,1,1)
        #control 0< Res_ini < GPP
        Res_ini = self.ReLU(annual_GPP+annual_Ra - annual_Yield)
        #Res_ini[Res_ini > annual_GPP].data = annual_GPP[Res_ini > annual_GPP].data 
        #scale Res_ini
        Res_ini = Z_norm_with_scaler(Res_ini,self.Res_scaler)
        ##calculate Rh now with current year res
        Res = Res_ini.repeat(1,self.seq_len,1)
        #left day 300
        Res[:,0:298,:] = 0.0
        Res[:,300:,:] = 0.0
        output1, hidden3  = self.gru_Rh(torch.cat((Res,self.drop(output),inputs), 2), hidden[2])
        Rh = self.densor_Rh(self.drop(output1))
        
        #NEE
        output1, hidden4 = self.gru_NEE(torch.cat((Ra,Rh,inputs), 2), hidden[3])
        output1 = torch.cat((Ra,Rh,self.densor_NEE(self.drop(output1))),2)


        
        return output1, output2, (hidden1,hidden2,hidden3,hidden4)
#bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(2, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(2, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))
    
# Simplified model for yield
class Yield_model_v12_1(nn.Module):
    def __init__(self, ninp, dropout, seq_len):
        super(Yield_model_v12_1, self).__init__()
        nhid = 64
        self.nhid = nhid
        self.seq_len = seq_len
        self.gru_basic = nn.GRU(ninp,nhid,2,dropout=dropout, batch_first=True)
        self.drop=nn.Dropout(dropout)
        #attn for yield prediction
        self.attn = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yield = nn.Sequential(
            nn.Linear(nhid, 64),
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
        for ii in range(4):
            self.attn[ii*2].bias.data.zero_()
            self.attn[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yield[ii*2].bias.data.zero_()
            self.densor_yield[ii*2].weight.data.uniform_(-initrange, initrange)
        

    def forward(self, inputs, hidden):
        output, hidden1 = self.gru_basic(inputs, hidden)
        #predict yield
        inputs2 = self.drop(output)
        attn_weights = F.softmax(self.attn(inputs2), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs2 = torch.bmm(attn_weights,inputs2)
        output2 = self.densor_yield(inputs2)

        return output2, hidden1
#bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.nhid)
    
# use Reco model v1--GRU model
class RecoGRU_flux(nn.Module):
    def __init__(self, ninp, nhid, nlayers, nout, dropout):
        super(RecoGRU_flux, self).__init__()
        if nlayers > 1:
            self.gru = nn.GRU(ninp, nhid,nlayers,dropout=dropout, batch_first=True)
        else:
            self.gru = nn.GRU(ninp, nhid,nlayers, batch_first=True)
        #self.densor1 = nn.ReLU() #can test other function
        self.densor2 = nn.Linear(nhid, nout)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 #may change to a small value
        self.densor2.bias.data.zero_()
        self.densor2.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        output, hidden = self.gru(inputs, hidden)
        #output = self.densor1(self.drop(output))
        #output = torch.exp(self.densor2(self.drop(output))) # add exp
        output = self.densor2(self.drop(output)) # add exp
        return output, hidden
#bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    

class RecoGRU_flux_v14(nn.Module):
    def __init__(self, ninp, nhid, nlayers, nout, dropout):
        super(RecoGRU_flux_v14, self).__init__()
        if nlayers > 1:
            self.gru = nn.GRU(ninp, nhid,nlayers,dropout=dropout, batch_first=True)
        else:
            self.gru = nn.GRU(ninp, nhid,nlayers, batch_first=True)
        #self.densor1 = nn.ReLU() #can test other function
        self.densor2 = nn.Linear(nhid, nout)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 #may change to a small value
        self.densor2.bias.data.zero_()
        self.densor2.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        output, hidden = self.gru(inputs, hidden)
        #output = self.densor1(self.drop(output))
        #output = torch.exp(self.densor2(self.drop(output))) # add exp
        output = self.densor2(self.drop(output)) # add exp
        ##########get two output so that we don't need to change others
        return output,output, hidden
#bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
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