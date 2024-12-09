import os
import torch
import numpy as np
import kgml_lib

# define functions from kgml_lib
Z_norm = kgml_lib.Z_norm 
Z_norm_reverse = kgml_lib.Z_norm_reverse

class DataSet:
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

    def load(self):
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

    # Don't need ,remove in future
    def process_step2_X(self):
		#=============Creating empty tensors to fill them with the acutal data=============
        # [365*years, 100, 19]
        self.X = torch.zeros([self.Tx * self.tyear, self.bsz, self.n_f]) #What's X? Tx * tyear = steps in time series? What's bsz and FIPS?
        # [19, 2] Each feature's mean and std values
        self.X_scaler = np.zeros([self.n_f, 2]) # what is this for?
        
        # fill features fts_1 ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP']
        fts_1_len = len(self.fts_names_1) # len is 9
        #self.X[:,:,:9] = self.data['X'][:,:,:9]
        self.X[:,:,0:fts_1_len] = self.data['X'][:,:,0:fts_1_len]

        # X_scaler has [mean, std] for 
        # ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP','Ra','Rh','GrainC']
        # last row is for 'SOC', but not used
        # Save 9 parms ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP']
        #self.X_scaler[:9,:] = self.data['X_scaler'][:9,:]
        self.X_scaler[0:fts_1_len,:] = self.data['X_scaler'][0:fts_1_len,:]

        # fill the feature "Year"
        for y in range(self.tyear):
            self.X[y * self.Tx :(y+1)*self.Tx, :, fts_1_len] = y + self.start
        
        # Get each year's Normalization value, 18 years' mean and std value
        # self.X[:,:,9], self.X_scaler[9,0], self.X_scaler[9,1] = self.Z_score(self.X[:,:,9])
        self.X[:,:,fts_1_len], self.X_scaler[fts_1_len,0], self.X_scaler[fts_1_len,1] = self.Z_score(self.X[:,:,fts_1_len])
        
        # fill features fsp: ['TBKDS','TSAND','TSILT','TFC','TWP','TKSat','TSOC','TPH','TCEC']
        for i in range(len(self.fsp)): # range(9)
            '''What does it do???'''
            # data['Xsp'] shape is [100, 9]
            # data['Xsp'][:,i] is a 1D tensor,shape is [100]
            # .view(1,self.bsz) conver to 2D tensor, shape is [1,100], 1 row, 100 columns
            # .repeat(self.Tx*self.tyear,1) repeat this row 365*18 times
            # each fsp feature's value is same for all years day
            # self.X[:,:,10+i] = .....
            self.X[:,:,(fts_1_len+1)+i] = self.data['Xsp'][:,i].view(1,self.bsz).repeat(self.Tx*self.tyear,1) # What's SP? Soin P..?
            self.X_scaler[(fts_1_len+1)+i,:] = self.data['Xsp_scaler'][i,:] 
        print(self.X.size())

    # Don't need, remove in future
    def process_step2_Y(self):
        # [365* years, 100, 3], 3 is len of ['Ra','Rh','NEE']
        self.Y1 = torch.zeros([self.Tx*self.tyear, self.bsz, self.n_out_1])
        #[3,2] each feature's mean and std value
        self.Y1_scaler = np.zeros([self.n_out_1, 2])

        # [years, 100, 1] 1 is the len of ['Yield']
        self.Y2 = torch.zeros([self.tyear, self.bsz, self.n_out_2])
        #[1,2]
        self.Y2_scaler = np.zeros([self.n_out_2, 2])

        # fts_1 = ['RADN','TMAX_AIR','TDIF_AIR','HMAX_AIR','HDIF_AIR','WIND','PRECN','Crop_Type','GPP']
        fts_1_len = len(self.fts_1) # len is 9
        len_Ra_Rh = 2
        GPP_index = self.fts_1.index('GPP') #8
        RA_index = self.fts_1.index('RADN') #0

        # Save ['Ra', 'Rh']
        # Y1_scaler[0:2,:] = data['X_scaler'][9:11,:]
        self.Y1_scaler[:len_Ra_Rh,:] = self.data['X_scaler'][fts_1_len:(fts_1_len + len_Ra_Rh),:]
        for i in range(len_Ra_Rh):
            self.Y1[:,:,i]= self.Z_score_inv(self.data['X'][:,:,fts_1_len+i],self.Y1_scaler[i,:],1.0)

        # GPP seems a local value, don't be a class attribute 
        self.GPP = self.Z_score_inv(self.X[:,:,GPP_index], self.X_scaler[GPP_index,:], 1.0)

        # self.Y1[:,:,2] = .... for set 'NEE' value
        self.Y1[:,:,len_Ra_Rh] = -(self.GPP+self.Y1[:,:,0]+self.Y1[:,:,1]) #????? GPP -Ra-Rh, Ra, Ra are negative,GPP +Ra+Rh+NEE = 0
        for i in range(self.n_out_1): #range(3)
            self.Y1[:,:,i], self.Y1_scaler[i,0], self.Y1_scaler[i,1] = self.Z_score(self.Y1[:,:,i])

        # Save ['GrainC']
        self.Y2_scaler[:,:] = self.data['X_scaler'][(fts_1_len + len_Ra_Rh),:] # (fts_1_len + len_Ra_Rh) is 11
        for y in range(self.tyear):
            self.Y2[y,:,0] = self.Z_score_inv(self.data['X'][(y+1)*self.Tx-2,:,(fts_1_len + len_Ra_Rh)], self.Y2_scaler[0,:], 1.0)
        self.Y2[:,:,0], self.Y2_scaler[0,0], self.Y2_scaler[0,1] = self.Z_score(self.Y2[:,:,0])


        GPP_annual_all = torch.zeros([self.tyear,self.bsz]) # shape is [18, 100]
        Ra_annual_all = torch.zeros([self.tyear,self.bsz]) # shape is [18, 100]

        for y in range(self.tyear):
            GPP_annual_all[y,:] = torch.sum(self.Z_score_inv(self.X[y*self.Tx:(y+1)*self.Tx,:,GPP_index],self.X_scaler[GPP_index,:],1.0),dim=0)
            Ra_annual_all[y,:] = torch.sum(self.Z_score_inv(self.Y1[y*self.Tx:(y+1)*self.Tx,:,RA_index],self.Y1_scaler[RA_index,:],1.0),dim=0)

        Res_annual_all = GPP_annual_all + Ra_annual_all - self.Z_score_inv(self.Y2[:,:,0],self.Y2_scaler[0,:],1.0)
        GPP_Res_f = torch.mean(GPP_annual_all,dim=0)/torch.mean(Res_annual_all,dim=0)

        self.GPP_Res_fmean = GPP_Res_f.mean()
        self.Res_scaler = np.zeros([1,2])

        #feature scaling of Res
        Res__, self.Res_scaler[0,0], self.Res_scaler[0,1] = self.Z_score(Res_annual_all)

        print(f"Y1 {self.Y1.size()}, Y2 {self.Y2.size()}")

    # random pickup n_years for each site as validation data
    # make sure for each row, have different values. Don't have case like: [12,12]
    def create_sample_index(self,n_site:int, sample_index_file="traindataset_split_year_v1.sav", 
                            n_years:int =2, random_state=42):
        #np.random.seed(random_state)

        train_n = n_site # 100 sites
        sample_index = np.zeros([train_n, n_years], dtype=int)

        _year_range = np.arange(0,self.tyear) # 18 years, value from 0-17
        for i in range(train_n):
            sample_index[i,:] = np.random.choice(_year_range, n_years, replace=False)

        torch.save({'sample_index':sample_index,
                    },self.data_path + sample_index_file)
        
        return sample_index


    # random choice n_years for each site as val dataset
    def train_test_split(self, batch_first=True, sample_index_file="traindataset_split_year_v1.sav", 
                        n_years:int =2, random_state=42):

        if batch_first:
            # convert shape: [years*365, 100, 19] -> [100, years*365, 19]
            self.X_train = torch.from_numpy(np.einsum('ijk->jik', self.X.numpy()))
            self.Y1_train = torch.from_numpy(np.einsum('ijk->jik', self.Y1.numpy()))

            # covert to [100, 18, 1] , 18 years
            self.Y2_train = torch.from_numpy(np.einsum('ijk->jik', self.Y2.numpy()))


        train_n = self.X_train.size(0) # train_n is 100
        #create mask for each data point. Random choose 2 year maske 0
        
        # check if file exist
        if not os.path.exists (self.data_path +  sample_index_file):  #'traindataset_split_year_v1.sav'
            sample_index = self.create_sample_index(train_n,sample_index_file, n_years, random_state)
        else:
            tmp = torch.load(self.data_path + sample_index_file, weights_only=False) #'traindataset_split_year_v1.sav'
            sample_index = tmp['sample_index']
            print(f"Load sample index file {self.data_path + sample_index_file}")

        # [100, years*365, 3], 18 years
        self.Y1_mask_train=torch.zeros(self.Y1_train.size())+1.0 # initail value 1
        self.Y1_mask_val=torch.zeros(self.Y1_train.size())   # initial value 0

        # [100, years, 1], 18 years
        self.Y2_mask_train=torch.zeros(self.Y2_train.size())+1.0
        self.Y2_mask_val=torch.zeros(self.Y2_train.size())

        # For each site, random pickup 2 years
        for i in range(train_n):
            for y in range(n_years):
                # set value for each day of one year
                self.Y1_mask_train[i,sample_index[i,y]*self.Tx:(sample_index[i,y]+1)*self.Tx,:] = 0.0
                self.Y1_mask_val[i,sample_index[i,y]*self.Tx:(sample_index[i,y]+1)*self.Tx,:] = 1.0
                # Set year value
                self.Y2_mask_train[i,sample_index[i,y],:] = 0.0
                self.Y2_mask_val[i,sample_index[i,y],:] = 1.0

        # ge(0.5) means value >= 0.5 as True
        self.Y1_maskb_val = self.Y1_mask_val.ge(0.5)
        self.Y2_maskb_val = self.Y2_mask_val.ge(0.5)

        # self.X_train = X_train
        # self.Y1_train = Y1_train
        # self.Y2_train = Y2_train

        # self.Y1_mask_train = Y1_mask_train
        # self.Y1_mask_val = Y1_mask_val
        # self.Y1_maskb_val = Y1_maskb_val

        # self.Y2_mask_train = Y2_mask_train
        # self.Y2_mask_val = Y2_mask_val
        # self.Y2_maskb_val = Y2_maskb_val



    # def Z_score(self, X: torch.Tensor) -> float:
    #     X_mean = X.numpy().mean(dtype=np.float64)
    #     X_std = np.std(np.array(X, dtype=np.float64))
    #     return (X - X_mean) / X_std, X_mean, X_std
        
    # def Z_score_inv(self, X: torch.Tensor, scaler: np.array, units_convert: float = 1.0) -> torch.Tensor:
    #     mean, std = scaler[0], scaler[1]
    #     return (X * std + mean) * units_convert