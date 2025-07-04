# PyKGML: Python library for knowledge-guided machine learning
------------------------------------------------------------

PyKGML [(GitHub repo)](https://github.com/AI4Agriculture/PyKGML_development) is a Python library to facilitate the development of knowledge-guided machine learning (KGML) models in natural and agricultural ecosystems. It aims to provide research and educational support with improved accessibility to ML-ready data and code for developing KGML models, testing new algorithms, and providing efficient model benchmarking.

# How to use PyKGML

### Data
The example datasets can be downloaded from Zenodo: https://doi.org/10.5281/zenodo.15580484   
It includes four files:
  - **co2_pretrain_data.sav**: the synthetic dataset of KGMLag-CO2.
  - **co2_finetune_data.sav**: the observation dataset of KGMLag-CO2.
  - **n2o_pretrain_data.sav**: the synthetic dataset of KGMLag-N2O.
  - **n2o_finetune_augment_data.sav**: the augmented observation dataset of KGMLag-N2O.

Each file is a serialized Python dictionary containing the following keys and values:

      data={'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'y_scaler': y_scaler,
            'input_features': input_features,
            'output_features': output_features} 

### Files 
  • **dataset.py** is used to prepare demo datasets for pretraining and fine-tuning: Step2_DataSet is the CO2 synthetic dataset for pretraining, and Step5_DataSet is the CO2 flux tower dataset for fine- 
  tuning. N2O_synthetic_dataset and N2O_mesocosm_dataset are the N2O pretraining and fine-tuning datasets, respectively.
  
  • **kgml_lib.py** contains utility functions such as for normalization (Z_norm) and computing coefficient of determination (R2Loss).
  
  • **time_series_models.py** assembles classes of model, and implement the processes of data preparation, model training and testing.
  
  • **unified_model_processing.ipynb** is a jupyter notebook to demonstrate the use of PyKGML.  
  

  The original code of KGMLag-CO2 and KGMLag-N2O and development code of PyKGML are stored in the folder "[development_materials](development_materials)":
    
  • KGMLag_CO2_pretrain_baseline.py and KGMLag_CO2_finetune_baseline.py are the original code from study 1.
    
  • KGMLag_N2O_pretrain_baseline.ipynb and KGMLag_N2O_finetune_baseline.ipynb are the original code from study 2.

  <!-- The baseline code were further modularized to some clean versions: 
    
    - KGMLag_CO2_pretrain_modularized.py are the modularized version of KGMLag_CO2_pretrain_baseline.py.
    
    - KGMLag_CO2_finetune_modularized.py are the modularized version of KGMLag_CO2_finetune_baseline.py.
    
    - KGMLag_N2O_modularized.py are the modularized version combining KGMLag_N2O_pretrain_baseline.py and KGMLag_N2O_finetune_baseline.py.
    
    - test_KGMLag_CO2_pretrain.ipynb is a testing call on KGMLag_CO2_pretrain_modularized.py
    
    - test_KGMLag_CO2_finetune.ipynb is a testing call on KGMLag_CO2_finetune_modularized.py -->

### Using PyKGML

**Import a model and the data preparer:**  

    from time_series_models import GRUSeq2SeqWithAttention, SequenceDataset

**Load and prepare data:**

    co2_finetune_file = data_path + 'co2_finetune_data.sav
    data = torch.load(co2_finetune_file, weights_only=False)
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']
    y_scaler = data['y_scaler']

    sequence_length = 365
    train_dataset = SequenceDataset(X_train, Y_train, sequence_length)
    test_dataset = SequenceDataset(X_test, Y_test, sequence_length)

**Model set up:**

    model = GRUSeq2SeqWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
    model.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    model.test_loader  = DataLoader(test_dataset, batch_size, shuffle=False)

    # set up hyperparameters:
    learning_rate = 0.001
    step_size = 40
    max_epoch = 200
    gamma = 0.8
    # loss function
    loss_function = nn.L1Loss()

**Model training and testing:**

    model.train_model(loss_fun=loss_function, LR=learning_rate, step_size=step_size, gamma=gamma, maxepoch=max_epoch)
    model.test()

More details about using PyKGML can be found in unified_model_processing.ipynb


# Benchmark dataset
We integrated data from two prior studies to support the functional pipeline of PyKGML. These studies represent pioneering work in agricultural knowledge-guided machine learning (KGML), advancing the simulation of agroecosystem dynamics related to greenhouse gas (GHG) fluxes:

[Study 1](https://doi.org/10.5194/gmd-15-2839-2022): Liu et al. (2022). KGML-ag: a modeling framework of knowledge-guided machine learning to simulate agroecosystems: a case study of estimating N2O emission using data from mesocosm experiments, Geosci. Model Dev., 15, 2839–2858.

[Study 2](https://www.nature.com/articles/s41467-023-43860-5): Liu et al. (2024). Knowledge-guided machine learning can improve carbon cycle quantification in agroecosystems. Nature communications, 15(1), 357.

Study 1 developed KGML models to predict N<sub>2</sub>O fluxes using *ecosys* synthetic data as the pretraining dataset and chamber observations in a mesocosm environment as the fine-tuning dataset. Study 2 developed KGML models for predicting and partitioning CO<sub>2</sub> fluxes (Ra, Rh), using synthetic data in the pretraining step and flux tower observations for fine-tuning. [***ecosys***](https://github.com/jinyun1tang/ECOSYS) is a process-based model that incoporates comprehensive ecosystem biogeochemical and biogeophysical processes into its modeling design. To differentiate KGML models from the two studies, we refer to the selected model of study 1 as KGMLag-CO2 and that of study 2 as KGMLag-N2O.

Two datasets were harmonized using the CO<sub>2</sub> flux dataset from study 1 and and the N<sub>2</sub>O flux dataset from study 2 to demonstrate the use of PyKGML:
1. **CO<sub>2</sub> dataset:**
  * Synthetic data of *ecosys*:
    - 100 simulations at random corn fields in the Midwest.
    - Daily sequences over 18 years (2000-2018).
  * Field observations:
    - Eddy-covariance observations from 11 flux towers in the Midwest.
    - A total of 102 years of daily sequences.
  * Input variables (19):
    - Meterological (7): solar radiation (RADN), max air T (TMAX_AIR), (max-min) air T (TDIF_AIR), max air humidity (HMAX_AIR), (max-min) air humidity (HDIF_AIR), wind speed (WIND), precipitation (PRECN).
    - Soil properties (9): bulk density (TBKDS), sand content (TSAND), silt content (TSILT), field capacity (TFC), wilting point (TWP), saturate hydraulic conductivity (TKSat), soil organic carbon concetration (TSOC), pH (TPH), cation exchange capacity (TCEC)
    - Other: crop type (Crop_Type), gross primary productivity (GPP)
  * Output variables (3):
    - Autotrophic respiration (Ra), heterotrophic respiration (Rh), carbon mass of grain (GrainC).
2. **N<sub>2</sub>O dataset:** 
  * Synthetic data of *ecosys*:
    - 1980 simulations at 99 counties x 20 N-fertilizer rates in the 3I states (Illinois, Iowa, Indiana).
    - Daily sequences over 18 years (2000-2018).
  * Field observations:
    - 6 chamber observations in a mesocosm environment facility at the University of Minnesota.
    - Daily sequences of 122 days x 3 years (2016-2017) x 1000 augmentations from hourly data at each chamber.
  * input variables (16):
    - Meterological (7): solar radiation (RADN), max air T (TMAX_AIR), min air T (TMIN_AIR), max air humidity (HMAX_AIR), min air humidity (HMIN_AIR), wind speed (WIND), precipitation (PRECN).
    - Soil properties (6): bulk density (TBKDS), sand content (TSAND), silt content (TSILT), pH (TPH), cation exchange capacity (TCEC), soil organic carbon concetration (TSOC)
    - Management (3): N-fertilizer rate (FERTZR_N), planting day of year (PDOY), crop type (PLANTT).
  * Output variables (3):
    - N<sub>2</sub>O FLUX (N2O_FLUX), soil CO<sub>2</sub> flux (CO2_FLUX), soil water content at 10 cm (WTR_3), soil ammonium concentration at 10 cm (NH4_3), soil nitrate concentration at 10 cm (NO3_3).


# PyKGML development
There are several strategies that can be leveraged to incoporate domain knowledge into the development of a KGML model, as explored and summarized in the two references ([Liu et al. 2022](https://doi.org/10.5194/gmd-15-2839-2022), [Liu et al. 2024](https://www.nature.com/articles/s41467-023-43860-5)):
1. Knowledge-guided initialization through pre-training with synthetic data generated generated by process-based models.
2. Knowledge-guided loss function design according to physical or chemical laws in ecosystems, such as mass balance, non-negative.
3. Hierarchical architecture design according to causal relations or adding dense layers containing domain knowledge.
4. Residual modeling with ML models to reduce the bias between PB model outputs and observations.
5. Other hybrid modeling approaches combining PB and ML models.  

In PyKGML, we aim to functionize strategies 1-3 in a user-friendly way, so that a user can develop a KGML model without compelx coding but by providing the design idea and data:
1. Pre-training strategy will be enabled as a model training step using provided data. 
2. Loss function design will be enabled in a way that converts a equation from the user to a constrained loss function in the ML model.
3. Architecture design will be enabled in a way that translate a user's idea of combining ML layers and blocks into functional code.

**In this current version of PyKGML, loss function design has been realized in a preliminary way, and the function of architecture design is still under development.**  We set loss function as a input parameter for model training so that users can select a popular loss function from other libraries, e.g. MAE loss or MSE loss in Pytorch, or define their own loss functions.

Models of KGMLag-CO2 and KGMLag-N2O were added to the model gallary of PyKGML so  users can adopt the tested architecture for training or fine-tuning. To make it clear, the KGMLag-CO2 and KGMLag-N2O models in PyKGML only refer to the architectures of the original models, not including the strategies involved in pretraining and fine-tuning steps to improve the model perforamnces. Instead, we generalize the process of model pre-training and fine-tuning for KGMLag-CO2, KGMLag-N2O, and pure ML models using the same datasets.

**Model gallary**:
  * Attention
  * GRUSeq2SeqWithAttention
  * GRUSeq2Seq
  * LSTMSeq2Seq
  * 1dCNN
  * RelPositionalEncoding
  * TimeSeriesTransformer
  * N2OGRU_KGML (This is the model architecture of KGMLag-N2O from [Liu et al., 2022](https://doi.org/10.5194/gmd-15-2839-2022))
  * RecoGRU_KGML (This is the model architecture of KGMLag-CO2 from [Liu et al., 2024](https://www.nature.com/articles/s41467-023-43860-5)) 


# Acknowledgement
Funding sources for this research includes:  
1. This research is part of [AI-LEAF: AI Institute for Land, Economy, Agriculture & Forestry](https://cse.umn.edu/aileaf) and is supported by USDA National Institute of Food and Agriculture (NIFA) and the National Science Foundation (NSF) National AI Research Institutes Competitive Award No. 2023-67021-39829.
2. The Forever Green Initiative of the University of Minnesota, using funds from the State of Minnesota Clean Water Fund provided by the Minnesota Department of Agriculture.  
3. National Science Foundation: Signal in the Soil program (award No. 2034385).
