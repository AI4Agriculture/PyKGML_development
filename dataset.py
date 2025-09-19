import os
import torch
import numpy as np
import kgml_lib
import matplotlib.pyplot as plt
import math

# define functions from kgml_lib
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
sample_data_FN = kgml_lib.sample_data_FN

class CO2_synthetic_dataset:
    '''
    data_path: input data directory
    input_data: input dataset file name
    out_path: output directory
    sample_index_file: random pickup two years for each site, this file store the index of selected years.
    '''
    def __init__(self,data_path: str, input_data:str, out_path: str, sample_index_file:str,
                 start: int = 2001, end: int =2018, Tx: int = 365, 
                 outNames_1: list = ['Ra','Rh','NEE'], outNames_2: list = ['Yield']) -> None:
        
        self.data_path = data_path # os.path.abspath(data_path)
        self.out_path = out_path # os.path.abspath(out_path)
        self.input_data = input_data
        self.sample_index_file = sample_index_file
        self.start = start
        self.end = end
        self.Tx = Tx
        self.tyear = end - start + 1
        self.outNames_1 = outNames_1
        self.outNames_2 = outNames_2
        self.n_out1 = len(outNames_1)
        self.n_out2 = len(outNames_2)
        self.fts_names_1 = ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP'] # 9 features
        self.fts_names_2 = ['Ra','Rh','GrainC'] # Not used
        #['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP','Ra','Rh','GrainC']
        self.fts_names = self.fts_names_1 + self.fts_names_2 # Not used 
        self.fsp_names = ['TBKDS','TSAND','TSILT','TFC','TWP','TKSat','TSOC','TPH','TCEC'] # 9 features
        self.f_names = self.fts_names_1 + ['Year']+ self.fsp_names # total 9+ 1+ 9 features
        self.n_f = len(self.f_names) # The number of features, now is 19 features

    def load_step2_data(self):
        self.data = torch.load(self.data_path + self.input_data, weights_only=False)
        # FIPS_ref is the unique code for each site
        # and FIPS is the county number being used in US
        self.bsz = len(self.data['FIPS_ref']) # Site IDs?

    def prepare_step2_data(self):
        Tx = self.Tx
        tyear = self.tyear
        bsz0 = self.bsz
        n_f = self.n_f
        data0 = self.data
        n_out1 = self.n_out1
        n_out2 = self.n_out2

        #initial input and output
        # scaler saves each feature's mean and std
        X=torch.zeros([Tx*tyear,bsz0,n_f]) #[365*18, 100, 19]
        X_scaler = np.zeros([n_f,2])  #[19, 2]
        Y1=torch.zeros([Tx*tyear,bsz0,n_out1]) #[365*18, 100, 3]
        Y1_scaler = np.zeros([n_out1,2]) #[3, 2]
        Y2=torch.zeros([tyear,bsz0,n_out2]) #[18, 100, 1]
        Y2_scaler = np.zeros([n_out2,2]) #[1, 2]

        #load in X variables
        # fill features ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP']
        fts_1_len = len(self.fts_names_1) # len is 9
        X[:,:,0:fts_1_len] = data0['X'][:,:,0:fts_1_len]
        X_scaler[0:fts_1_len,:] = data0['X_scaler'][0:fts_1_len,:]
        
        #fill the feature "Year"
        for y in range(tyear):
            X[y*Tx:(y+1)*Tx,:,fts_1_len] = y+self.start
        
        #range0=0
        #range1=1
        # Get each year's Normalization value, 18 years' mean and std value
        # self.X[:,:,9], self.X_scaler[9,0], self.X_scaler[9,1] = self.Z_score(self.X[:,:,9])
        X[:,:,fts_1_len], X_scaler[fts_1_len,0], X_scaler[fts_1_len,1] = Z_norm(X[:,:,fts_1_len])

        # fill fsp 9 features：['TBKDS','TSAND','TSILT','TFC','TWP','TKSat','TSOC','TPH','TCEC']
        for i in range(len(self.fsp_names)):
            # data['Xsp'] shape is [100, 9]
            # data['Xsp'][:,i] is a 1D tensor,shape is [100]
            # .view(1,self.bsz) conver to 2D tensor, shape is [1,100], 1 row, 100 columns
            # .repeat(self.Tx*self.tyear,1) repeat this row 365*18 times
            # each fsp feature's value is same for all years day and all sites
            # self.X[:,:,10+i] = .....
            X[:,:,(fts_1_len+1)+i] = data0['Xsp'][:,i].view(1,bsz0).repeat(Tx*tyear,1)
            X_scaler[(fts_1_len+1)+i,:] = data0['Xsp_scaler'][i,:]
            
        len_Ra_Rh = 2
        GPP_index = self.fts_names_1.index('GPP') #8
        RA_index = self.fts_names_1.index('RADN') #0

        #load in Y1
        # Save ['Ra', 'Rh']
        Y1_scaler[0:2,:] = data0['X_scaler'][9:11,:]
        #Y1_scaler[0:len_Ra_Rh,:] = data0['X_scaler'][fts_1_len:(fts_1_len + len_Ra_Rh),:]
        for i in range(2):
            Y1[:,:,i]= Z_norm_reverse(data0['X'][:,:,9+i],Y1_scaler[i,:],1.0)
            #Y1[:,:,i]= Z_norm_reverse(data0['X'][:,:,fts_1_len+i],Y1_scaler[i,:],1.0)
        GPP = Z_norm_reverse(X[:,:,8],X_scaler[8,:],1.0)
        #GPP = Z_norm_reverse(X[:,:,GPP_index],X_scaler[GPP_index,:],1.0)
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
        print(X.size(),Y1.size(),Y2.size())
        print(self.f_names)

        self.X = X
        self.X_scaler = X_scaler
        self.Y1 = Y1
        self.Y1_scaler = Y1_scaler
        self.Y2= Y2
        self.Y2_scaler = Y2_scaler

        self.GPP_Res_fmean = GPP_Res_fmean
        self.Res_scaler = Res_scaler

class CO2_fluxtower_dataset:
    '''
    data_path: input data directory
    input_data: input dataset file name
    out_path: output directory
    sample_index_file: random pickup two years for each site, this file store the index of selected years.
    '''
    # sample_index_file:str,
    def __init__(self,data_path: str, out_path: str,
                 start: int = 2001, end: int =2018, Tx: int = 365, 
                 outNames_1: list = ['Ra','Rh','NEE'], outNames_2: list = ['Yield']) -> None:
        
        self.data_path = data_path
        self.out_path = out_path

        #self.sample_index_file = sample_index_file
        self.start = start
        self.end = end
        self.Tx = Tx
        self.tyear = end - start + 1
        self.outNames_1 = outNames_1
        self.outNames_2 = outNames_2
        self.n_out1 = len(outNames_1)
        self.n_out2 = len(outNames_2)
        self.fts_names_1 = ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP'] # 9 features
        self.fts_names_2 = ['Ra','Rh','GrainC'] # Not used
        #['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP','Ra','Rh','GrainC']
        self.fts_names = self.fts_names_1 + self.fts_names_2 # Not used 
        self.fsp_names = ['TBKDS','TSAND','TSILT','TFC','TWP','TKSat','TSOC','TPH','TCEC'] # 9 features
        self.f_names = self.fts_names_1 + ['Year']+ self.fsp_names # total 9+ 1+ 9 features
        self.n_f = len(self.f_names) # The number of features, now is 19 features

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_scaler_data(self, scaler_file):
        data0 = torch.load(self.data_path + scaler_file, weights_only=False)
        
        self.X_scaler = data0['X_scaler']
        self.Y1_scaler = data0['Y1_scaler']
        self.Y2_scaler = data0['Y2_scaler']
        self.Res_scaler = data0['Res_scaler']
        self.GPP_Res_fmean = data0['GPP_Res_fmean']

        self.data = data0

    # file: 'fluxtower_inputs_noscale_v2.sav'
    def load_fluxtower_inputs_data(self, fluxtower_input_file):
        path_load = self.data_path + fluxtower_input_file
        data0=torch.load(path_load, weights_only=False)
        self.mode_input = data0['mode_input']
    
    # file: 'fluxtower_observe_noscale_v2.sav'
    def load_fluxtower_observe_data(self, fluxtower_observe_file): 
        path_load = self.data_path + fluxtower_observe_file
        data0=torch.load(path_load, weights_only=False)
        self.org_Reco = data0['org_Reco']
        self.org_NEE = data0['org_NEE']

    # create sample index and mask for train and validation
    def create_sample_index(self,sample_index_file="flux_split_year_v1.sav", n_years:int =2):

        sample_indexes = []
        for i in range(len(self.mode_input)):
            tyear1 = self.org_Reco[i].shape[0] # total years
            # Random pickup two years from tyear1
            sample_index = np.random.randint(tyear1, size=n_years)
            sample_indexes.append(sample_index)
                
        if not os.path.exists (self.out_path + sample_index_file):
            torch.save({'sample_indexes':sample_indexes,
                    },self.out_path + sample_index_file)
            
        return sample_indexes

    def prepare_data(self, sample_index_file):

        device = self.device

        Y_Reco_obs = []
        Y_NEE_obs = []
        X_sites = []
        Y_masks = []
        Y_masksb = []
        Y_masks_train = []
        Y_masks_val = []
        #sample_indexes = []
        if os.path.exists (self.out_path + sample_index_file):
            tmp = torch.load(self.out_path + sample_index_file,weights_only=False)
            sample_indexes = tmp['sample_indexes']
        else:
            sample_indexes = self.create_sample_index(sample_index_file, n_years=2)
        
        
        for i in range(len(self.mode_input)):
            tyear1 = self.org_Reco[i].shape[0] # total years
            totsq1= self.mode_input[i].size(1) # 365* total years
            #1) reshape the observed data to predicted scale
            Y_Reco_obs_t = np.zeros(totsq1)
            Y_NEE_obs_t = np.zeros(totsq1)
            Y_mask_t = np.zeros(totsq1) + 1.0
            for y in range(tyear1):
                Y_Reco_obs_t[y*365:(y+1)*365] = self.org_Reco[i][y,0:365]
                Y_NEE_obs_t[y*365:(y+1)*365]= self.org_NEE[i][y,0:365]
            
            #replace nan to 0 to avoid error
            nanindex = np.logical_or(np.isnan(Y_Reco_obs_t),np.isnan(Y_NEE_obs_t))
            Y_Reco_obs_t[nanindex] = 0.0 
            Y_NEE_obs_t[nanindex] = 0.0
            Y_mask_t[nanindex] = 0.0
            GPP_obs_t = Y_Reco_obs_t - Y_NEE_obs_t
            GPP_obs_t[nanindex] = self.mode_input[i][0,nanindex,8].numpy()
            nanindex = np.isnan(GPP_obs_t)
            GPP_obs_t[nanindex] = 0.0
            Y_mask_t[nanindex] = 0.0
            #corrected GPP
            self.mode_input[i][0,:,8] = torch.from_numpy(GPP_obs_t)
            Y_Reco_obs.append(torch.from_numpy(Y_Reco_obs_t).to(device))
            Y_NEE_obs.append(torch.from_numpy(Y_NEE_obs_t).to(device))
            Y_masks.append(torch.from_numpy(Y_mask_t).to(device))
            
            #2) scale th org model input
            X_site = torch.zeros(self.mode_input[i].size())
            for f in range(len(self.f_names)):
                X_site[:,:,f] = Z_norm_with_scaler(self.mode_input[i][:,:,f],self.X_scaler[f,:])
            X_sites.append(X_site.to(device))
            print(Y_masks[i].shape,X_sites[i].shape)
            #print(Y_masks[i].size(),X_sites[i].size())

            #######develop sample index and mask for train and validation
            # sample_index = np.random.randint(tyear1, size=(2))
            # if len(sample_indexes) > i:
            #     sample_index = sample_indexes[i]
            # else:
            #     sample_indexes.append(sample_index)
            
            sample_index = sample_indexes[i]
            Y_mask_train=torch.zeros(Y_masks[i].shape)+1.0
            Y_mask_val=torch.zeros(Y_masks[i].shape)
            for yy in range(2):
                Y_mask_train[sample_index[yy]*365:(sample_index[yy]+1)*365] = 0.0
                Y_mask_val[sample_index[yy]*365:(sample_index[yy]+1)*365] = 1.0
            Y_masks_train.append(Y_mask_train.to(device)*Y_masks[i])
            Y_masks_val.append(Y_mask_val.to(device)*Y_masks[i])

        self.X_sites = X_sites
        self.Y_Reco_obs = Y_Reco_obs
        self.Y_NEE_obs = Y_NEE_obs
        self.Y_masks_train = Y_masks_train
        self.Y_masks_val = Y_masks_val



class N2O_synthetic_dataset:
        
    def __init__(self,data_path: str, scaler_path:str, out_path:str, 
                 start: int = 2001, end: int =2018, Tx: int = 365,
                 outNames_1: list = ['CO2_FLUX', 'WTR_3','NH4_3', 'NO3_3'], outNames_2: list = ['N2O_FLUX']) -> None:
        '''
        data_path: input data directory
        input_data: input dataset file name
        out_path: output directory
        '''
        self.data_path = data_path # os.path.abspath(data_path)
        self.out_path = out_path # os.path.abspath(out_path)
        # self.input_data = data_path + input_data
        self.scaler_path = scaler_path
        self.fts_names = ['FERTZR_N','RADN','TMAX_AIR','TMIN_AIR','HMAX_AIR','HMIN_AIR','WIND','PRECN'] # 8 weather features
        self.fsp_names = ['PDOY', 'PLANTT', 'TBKDS', 'TCSAND', 'TCSILT', 'TPH', 'TCEC', 'TSOC'] # 2 management + 6 soil features
        self.f_names = self.fts_names + self.fsp_names # total 16 features
        self.out_names = outNames_2 + outNames_1 # target 'N2O_FLUX' and 4 intermediate variables 
        self.n_f = len(self.f_names) # The number of features
        self.n_out = len(self.out_names)
        self.n_out1 = self.n_out - 1
        self.n_out2 = self.n_out - self.n_out1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def load_data(self):
        self.data = torch.tensor(torch.load(self.data_path, weights_only=False)) # test with a smaller dataset [:,:1000,:]
        if self.scaler_path:
            self.scalers = torch.tensor(torch.load(self.scaler_path, weights_only=False))
        else:
            self.scalers = None
            print('Scalers of the dataset are not provided, will implement normalization in the step of prepare_data()')
        self.batch_total = self.data.size()[1] # site sample size in the 3 dimension data [DOY, site, features]
        print('data size is ', self.data.size())
    
    def prepare_data(self):
        if self.scalers is None or self.scalers.size == 0:
            self.scalers = np.zeros((self.data.shape[-1], 2))
            for i in range(self.data.shape[-1]):
                self.data[:,:,i], self.scalers[i, 0], self.scalers[i, 1] = Z_norm(self.data[:,:,i])

        self.X = self.data[:,:,:16]
        self.Y = self.data[:,:,16:]
        self.Xscaler = self.scalers[:16, :]
        self.Yscaler = self.scalers[16:, :]
        # print('Yscaler', self.Yscaler)
        print('Inputs include ', self.f_names)
        print('Outputs include ', self.out_names)
        
    # not in use, can be removed later    
    def train_test_split(self, train_ratio, test_ratio, val_ratio_to_train):
        if train_ratio and not test_ratio:
            test_ratio = 1 - train_ratio
        if not train_ratio and test_ratio:
            train_ratio = 1 - test_ratio
        if not val_ratio_to_train:
            val_ratio_to_train = 0.1
        # shuffle along site-year dimension. This randomization may be different every time
        shuffled_ix = torch.randperm(self.X.size()[1])
        self.X = self.X[:,shuffled_ix,:].to(self.device)
        self.Y = self.Y[:,shuffled_ix,:].to(self.device)
        
        self.total_n=self.X.size()[1]
        self.train_n=int((self.total_n * train_ratio) * (1-val_ratio_to_train))
        self.val_n=int((self.total_n * train_ratio) - self.train_n)
        self.test_n=self.total_n - self.train_n - self.val_n
        
        self.X_train=self.X[:,:self.train_n,:]
        self.X_val=self.X[:,self.train_n:(self.train_n+self.val_n),:]
        self.X_test=self.X[:,(self.train_n+self.val_n):,:]
        self.Y_train=self.Y[:,:self.train_n,:]
        self.Y_val=self.Y[:,self.train_n:(self.train_n+self.val_n),:]
        self.Y_test=self.Y[:,(self.train_n+self.val_n):,:]

        #sample the training data with sliding window
        # self.X_train_new, self.Y_train_new = sample_data_FN(self.X_train, self.Y_train, fn_ind=0)
        # self.X_val_new, self.Y_val_new = sample_data_FN(self.X_val, self.Y_val, fn_ind=0)
        # self.X_test_new, self.Y_test_new = sample_data_FN(self.X_test, self.Y_test, fn_ind=0)
        # self.X_train_new, self.Y_train_new = self.X_train, self.Y_train
        # self.X_val_new, self.Y_val_new = self.X_val, self.Y_val
        # self.X_test_new, self.Y_test_new = self.X_test, self.Y_test



class N2O_mesocosm_dataset:

    def __init__(self,data_path: str, out_path:str, 
                 start: int = 2016, end: int =2018, Tx: int = 365, scaler_path:str = None,
                 outNames_1: list = ['CO2_FLUX', 'WTR_3','NH4_3', 'NO3_3'], outNames_2: list = ['N2O_FLUX']) -> None:
        '''
        data_path: input data directory
        input_data: input dataset file name
        out_path: output directory
        '''
        self.data_path = data_path # os.path.abspath(data_path)
        self.out_path = out_path # os.path.abspath(out_path)
        # self.input_data = data_path + input_data
        self.scaler_path = scaler_path
        self.fts_names = ['FERTZR_N','RADN','TMAX_AIR','TMIN_AIR','HMAX_AIR','HMIN_AIR','WIND','PRECN'] # 8 weather features
        self.fsp_names = ['PDOY', 'PLANTT', 'TBKDS', 'TCSAND', 'TCSILT', 'TPH', 'TCEC', 'TSOC'] # 2 management + 6 soil features
        self.f_names = self.fts_names + self.fsp_names # total 16 features
        self.out_names = outNames_2 + outNames_1 # target 'N2O_FLUX' and 4 intermediate variables 
        self.n_f = len(self.f_names) # The number of features
        self.n_out = len(self.out_names)
        self.n_out1 = self.n_out - 1
        self.n_out2 = self.n_out - self.n_out1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def load_data(self):
        self.data = torch.tensor(torch.load(self.input_data, weights_only=False)) # test with a smaller dataset [:,:1000,:]
        if self.scaler_path:
            self.scalers = torch.tensor(torch.load(self.scaler_data, weights_only=False))
        else:
            print('Please provide scalers for the fine tune dataset')
        self.batch_total = self.data.size()[1] # site sample size in the 3 dimension data [DOY, site, features]
        print('data size is ', self.data.size())

    def prepare_data(self):
        self.X = self.data[:,:,:16]
        self.Y = self.data[:,:,16:]
        self.Xscaler = self.scalers[:16, :]
        self.Yscaler = self.scalers[16:, :]

    # fine tune data augmentation: 
    # 16 h of data are randomly selected from 24 h observations to compute their mean as the daily value.
    # The total number of data is augmented to 122 d × 3 years × 6 chambers × 1000 data samples in this study.
    # X_train, Y_train are augmented outputs; X_train_d, Y_train_d are outputs without augmentation.
    def augment_finetune_data(self, dataset_path, scaler_path, val_chamber:list = [], augment_factor: int = None):
        if not augment_factor:
            augn = 1000
        else:
            augn = augment_factor
        data0=torch.load(dataset_path)
        scalers = torch.load(scaler_path)
        Xscaler = self.Xscaler
        Yscaler = self.Yscaler
        X1=data0['InputX1']
        X2=data0['InputX2']
        X3=data0['Soil_p']
        Y=data0['OutputY']
        X1names = ['tair','swdown','precip','spRH'] 
        X2names = ['Obs_prec','Fertilizer']
        X3names=['TSN','FBCU','PDOY','PDS','PDD','DDOY','PLANTT',\
                  'LAT','TLB','TBKDS', 'TCSAND', 'TCSILT', 'TPH', 'TCEC', 'TSOC']
        Ynames= ['N2O_FLUX','CO2_FLUX','NO3','NH4','WFPS']
        days=122
        nyear=3
        totnchamber=6
        if len(val_chamber) == 0:
            val_chamber.append(np.random.randint(0, 6))
        print(f"chamber {val_chamber} is used to validate")
        c_index = [0,1,2,3,4,5]
        c_val = val_chamber
        nc_val = len(c_val)
        c_train = c_index
        for num in c_val:
            c_train.remove(num)
        nc_train = len(c_train)
        print(X1.shape,X2.shape,X3.shape,Y.shape)
        pred_names=['N2O_FLUX','CO2_FLUX','NO3_3','NH4_3','WTR_3']
        #load data n
        Ynames_n = [0,1,2,3,4]
        #find the pred_names number in out_names, 
        #the no. of model output Y_train_pred[pred_names_n[i]] will be the related to Y_train[Ynames_n[i]]
        pred_names_n = []
        for i in range(len(pred_names)):
            pred_names_n.append(self.out_names.index(pred_names[i]))
            
        Y_units_convert=[-24.0,-24.0,1.0,1.0,(1-1.5/2.65)/100.0]
    
        X_train = np.zeros([days,augn*nyear*nc_train,len(self.f_names)],dtype=np.float32)
        Y_train = np.zeros([days,augn*nyear*nc_train,len(Ynames)],dtype=np.float32)
        Y_train_mask = np.zeros(Y_train.shape,dtype=np.float32)
        #for training without augmentation
        X_train_d = np.zeros([days,nyear*nc_train,len(self.f_names)],dtype=np.float32)
        Y_train_d = np.zeros([days,nyear*nc_train,len(Ynames)],dtype=np.float32)
        Y_train_d_mask = np.zeros(Y_train_d.shape,dtype=np.float32) 
        
        X_val=np.zeros([days,nyear*nc_val,len(self.f_names)],dtype=np.float32)
        Y_val=np.zeros([days,nyear*nc_val,len(Ynames)],dtype=np.float32)
        Y_val_mask=np.zeros(Y_val.shape,dtype=np.float32) 
    
        
        #Y_gt ground truth first day index, for initials creating
        Y_train_gt_1stind = np.zeros([nyear*nc_train,len(Ynames)], dtype=int)
        Y_val_gt_1stind = np.zeros([nyear*nc_val,len(Ynames)], dtype=int)
        print(Y_train_gt_1stind.shape,Y_val_gt_1stind.shape)
        #Method: Multidimensional Shifting using NumPy
        #method from https://ethankoch.medium.com/incredibly-fast-random-sampling-in-python-baf154bd836a
        #product index_array (num_samples,sample_size) within elements
        # constants
        # returning index
        num_samples = augn
        sample_size = 16 #sample 16 hours within one day
        num_elements = 24
        #elements = np.arange(num_elements)
        # probabilities should sum to 1
        probabilities = np.random.random(num_elements)
        probabilities /= np.sum(probabilities)
        def multidimensional_shifting(num_samples, sample_size, probabilities):
            # replicate probabilities as many times as `num_samples`
            replicated_probabilities = np.tile(probabilities, (num_samples, 1))
            # get random shifting numbers & scale them correctly
            random_shifts = np.random.random(replicated_probabilities.shape)
            random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
            # shift by numbers & find largest (by finding the smallest of the negative)
            shifted_probabilities = random_shifts - replicated_probabilities
            return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]
    
        #sample data from mesocosm site chambers
        for d in range(days):
            #for training data with data augmentation
            for y in range(nyear):
                for c in range(nc_train):
                    #get random sampled indexes
                    sample_indexes = multidimensional_shifting(num_samples, sample_size, probabilities)
                    #input data
                    #temperature
                    elements = np.tile(X1[d*24:(d+1)*24,y,c_train[c],0], (num_samples, 1)) # copy the hourly data num_samples times
                    output_samples = np.take_along_axis(elements, sample_indexes, axis=1) # sample the data based on random indexes
                    output_samples_tmax = output_samples.max(1)
                    output_samples_tdif = output_samples_tmax-output_samples.min(1)
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),2]=output_samples_tmax
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),3]=output_samples_tdif
                    X_train_d[d,y*nc_train+c,2] = np.max(X1[d*24:(d+1)*24,y,c_train[c],0])
                    X_train_d[d,y*nc_train+c,3] = np.max(X1[d*24:(d+1)*24,y,c_train[c],0])-\
                                                            np.min(X1[d*24:(d+1)*24,y,c_train[c],0])
                    #radiation need to convert from W/m-2 to MJ m-2 d-1, *3600*24*10-6
                    elements = np.tile(X1[d*24:(d+1)*24,y,c_train[c],1], (num_samples, 1)) # copy the hourly data num_samples times
                    output_samples = np.take_along_axis(elements, sample_indexes, axis=1) # sample the data based on random indexes
                    output_samples_rad = output_samples.mean(1)*(3600.0*24.0*(10**(-6)))
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),1]=output_samples_rad
                    X_train_d[d,y*nc_train+c,1] = np.mean(X1[d*24:(d+1)*24,y,c_train[c],1])*(3600.0*24.0*(10**(-6)))
                    #humidity
                    elements = np.tile(X1[d*24:(d+1)*24,y,c_train[c],3], (num_samples, 1)) # copy the hourly data num_samples times
                    output_samples = np.take_along_axis(elements, sample_indexes, axis=1) # sample the data based on random indexes
                    output_samples_hmax = output_samples.max(1)
                    output_samples_hdif = output_samples_hmax - output_samples.min(1)
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),4]=output_samples_hmax
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),5]=output_samples_hdif
                    X_train_d[d,y*nc_train+c,4] = np.max(X1[d*24:(d+1)*24,y,c_train[c],3])
                    X_train_d[d,y*nc_train+c,5] = np.max(X1[d*24:(d+1)*24,y,c_train[c],3]) - \
                                                            np.min(X1[d*24:(d+1)*24,y,c_train[c],3])
                    #sample Y data
                    for ffy in range(len(Ynames)): 
                        element=Y[d*24:(d+1)*24,y,c_train[c],ffy]
                        nan_nums=np.count_nonzero(np.isnan(element))
                        if  nan_nums < 16:
                            # copy the hourly data num_samples times
                            elements = np.tile(element, (num_samples, 1)) 
                            # sample the data based on random indexes
                            output_samples = np.take_along_axis(elements, sample_indexes, axis=1) 
                            #convert to right units (n2O g N m-2 h-1 to d-1)
                            output_samples_n2o = np.nanmean(output_samples,axis=1)
                            # need to be direction to soil
                            Y_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),ffy]= output_samples_n2o*Y_units_convert[ffy]
                            Y_train_mask[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),ffy] = (24.0-float(nan_nums))/24.0
                            
                            Y_train_d[d,y*nc_train+c,ffy]= np.nanmean(Y[d*24:(d+1)*24,y,c_train[c],ffy])*\
                                                            Y_units_convert[ffy] #convert 
                            Y_train_d_mask[d,y*nc_train+c,ffy] = (24.0-float(nan_nums))/24.0
                            #get the first day of ground truth
                            if Y_train_gt_1stind[y*nc_train+c,ffy] == 0:
                                Y_train_gt_1stind[y*nc_train+c,ffy] = d
    
                        else:
                            # if missing value >=16, we use -999 represent nan
                            Y_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),ffy]=-999.0 
                            Y_train_mask[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),ffy] = 0.0
                            
                            Y_train_d[d,y*nc_train+c,ffy]=-999.0 
                            Y_train_d_mask[d,y*nc_train+c,ffy] = 0.0
                    #deal with other training variables
                    #fertilizer
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),0] = X2[d,y,c_train[c],1]
                    X_train_d[d,y*nc_train+c,0] = X2[d,y,c_train[c],1]
                    #wind
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),6] = 0.05
                    X_train_d[d,y*nc_train+c,6] = 0.05
                    #precipitation
                    X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),7] = X2[d,y,c_train[c],0]
                    X_train_d[d,y*nc_train+c,7] = X2[d,y,c_train[c],0]
                    for i in range(len(self.fsp_names)):
                        X_train[d,augn*(y*nc_train+c):augn*(y*nc_train+c+1),8+i] = X3[d,y,c_train[c],X3names.index(self.fsp_names[i])]
                        X_train_d[d,y*nc_train+c,8+i] = X3[d,y,c_train[c],X3names.index(self.fsp_names[i])]
    
    
        #load the validation data
        for d in range(days):
            for y in range(nyear):
                for c in range(nc_val):
                    #temperature
                    X_val[d,y*nc_val+c,2] = np.max(X1[d*24:(d+1)*24,y,c_val[c],0])
                    X_val[d,y*nc_val+c,3] = np.max(X1[d*24:(d+1)*24,y,c_val[c],0])-\
                                                            np.min(X1[d*24:(d+1)*24,y,c_val[c],0])
                    #radiation
                    X_val[d,y*nc_val+c,1] = np.mean(X1[d*24:(d+1)*24,y,c_val[c],1])*(3600.0*24.0*(10**(-6)))
                    #humidity
                    X_val[d,y*nc_val+c,4] = np.max(X1[d*24:(d+1)*24,y,c_val[c],3])
                    X_val[d,y*nc_val+c,5] = np.max(X1[d*24:(d+1)*24,y,c_val[c],3]) - \
                                                            np.min(X1[d*24:(d+1)*24,y,c_val[c],3])
                    #Y data
                    for ffy in range(len(Ynames)): 
                        element = Y[d*24:(d+1)*24,y,c_val[c],ffy]
                        nan_nums=np.count_nonzero(np.isnan(element))
                        if  nan_nums < 16:
                            Y_val[d,y*nc_val+c,ffy] = np.nanmean(element)*Y_units_convert[ffy] #convert 
                            Y_val_mask[d,y*nc_val+c,ffy] = (24.0-float(nan_nums))/24.0
                            #get the first day of ground truth
                            if Y_val_gt_1stind[y*nc_val+c,ffy] == 0:
                                Y_val_gt_1stind[y*nc_val+c,ffy] = d
                        else:
                            Y_val[d,y*nc_val+c,ffy] = -999.0 # if missing value >=16, we use -999 represent nan
                            Y_val_mask[d,y*nc_val+c,ffy] = 0.0
                    #deal with other training variables
                    #fertilizer
                    X_val[d,y*nc_val+c,0] = X2[d,y,c_val[c],1]
                    #wind
                    X_val[d,y*nc_val+c,6] = 0.05
                    #precipitation
                    X_val[d,y*nc_val+c,7] = X2[d,y,c_val[c],0]
                    for i in range(len(self.fsp_names)):
                        X_val[d,y*nc_val+c,8+i] = X3[d,y,c_val[c],X3names.index(self.fsp_names[i])]
    
        print(X_train.shape,Y_train.shape,X_train_d.shape,Y_train_d.shape,X_val.shape,Y_val.shape)
        # print(Xscaler.shape, Yscaler.shape,X_train.shape[2])
        #Z-norm the matrix
        for i in range(X_train.shape[2]):
            X_train[:,:,i]=Z_norm_with_scaler(X_train[:,:,i],Xscaler[i,:])
            X_train_d[:,:,i]=Z_norm_with_scaler(X_train_d[:,:,i],Xscaler[i,:])
            X_val[:,:,i]=Z_norm_with_scaler(X_val[:,:,i],Xscaler[i,:])
        for i in range(len(Ynames_n)):
            Y_train[:,:,Ynames_n[i]]=Z_norm_with_scaler(Y_train[:,:,Ynames_n[i]],Yscaler[pred_names_n[i],:])
            Y_train_d[:,:,Ynames_n[i]]=Z_norm_with_scaler(Y_train_d[:,:,Ynames_n[i]],Yscaler[pred_names_n[i],:])
            Y_val[:,:,Ynames_n[i]]=Z_norm_with_scaler(Y_val[:,:,Ynames_n[i]],Yscaler[pred_names_n[i],:])

        self.augn = augn
        self.c_val = c_val
        self.val_chamber = val_chamber
        self.aug_X_train = X_train
        self.aug_Y_train = Y_train
        self.aug_X_train_d = X_train_d
        self.aug_Y_train_d = Y_train_d
        self.aug_X_val = X_val
        self.aug_Y_val = Y_val
        self.aug_Y_train_mask = Y_train_mask
        self.aug_Y_val_mask = Y_val_mask
        self.aug_Y_train_d_mask = Y_train_d_mask
        self.aug_Y_train_gt_1stind = Y_val_gt_1stind
        self.aug_Y_val_gt_1stind = Y_val_gt_1stind

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