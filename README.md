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

Each file is a serialized Python dictionary containing the following keys and values, except that co2_finetune_data has no y_scaler because its Y_train and Y_test are not standardized:

      data={'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'input_features': input_features,
            'output_features': output_features} 

### Files 

  • **Tutorial_CO2_Colab.ipynb** is a Jupyter Notebook tutorial that runs on Google Colab. This is recommended for new users who are unfamiliar with Python or do not have Jupyter installed locally.    

  • **Tutorial_CO2_local.ipynb** is a Jupyter Notebook tutorial for local demonstration and requires a Python environment with Jupyter installed. 

  • **time_series_models.py** defines model classes and the processes for data preparation, model training and testing. 

  • **dataset.py** is used to prepare the example datasets: 'CO2_synthetic_dataset' generates the CO2 pretraining dataset and 'CO2_fluxtower_dataset' generates the CO2 fine-tuning dataset. 'N2O_synthetic_dataset' and 'N2O_mesocosm_dataset' prepare the N2O pretraining dataset and the N2O fine-tuning dataset, respectively. 
  
  • **kgml_lib.py** defines utility functions such as normalization (Z_norm) and  coefficient of determination computation (R2Loss).

  The development materials of PyKGML including original code of KGMLag-CO2 and KGMLag-N2O are stored in the folder [development_materials](development_materials).


### Using PyKGML

**Environment**  
We use Jupyter Notebook ([try it online or install locally](https://docs.jupyter.org/en/stable/start/)) for Python to example PyKGML usage on both cloud and local environments:  
  1. **Google Colab** (recommended for new users): is a hosted Jupyter Notebook service that requires no setup to use and provides free access to computing resources including GPUs. To get started with Google Colab, please refer to [Colab's official tutorial](https://colab.research.google.com/). The Colab notebook on PyKGML demonstration is [Tutorial_CO2_Colab.ipynb](Tutorial_CO2_Colab.ipynb).

  2. **Local** (or other cloud computing platform): The notebook on local PyKGML demonstration is [Tutorial_CO2_local.ipynb](Tutorial_CO2_local.ipynb). To use this notebook, The following applications and packages are required:  
      - Python 3 ([download](https://www.python.org/downloads/))  
      - Jyputer Notebook ([installation](https://docs.jupyter.org/en/stable/install/notebook-classic.html)).  
      - Python packages ([installation guidance](https://packaging.python.org/en/latest/tutorials/installing-packages/)):  
        - numpy  
        - pandas  
        - torch  
        - subprocess  
        - ast  
        - collections  
        - re  
        - matplotlib  
        - sklearn  
        - scipy  
        - inspect  

****
**Get started** with the PyKGML tutorial, [Tutorial_CO2_Colab.ipynb](Tutorial_CO2_Colab.ipynb) or [Tutorial_CO2_local.ipynb](Tutorial_CO2_local.ipynb).  

**Import a model and the data preparer:**  

    from time_series_models import GRUSeq2SeqWithAttention, SequenceDataset

**Load and prepare data:**

    co2_finetune_file = data_path + 'co2_finetune_data.sav'
    data = torch.load(co2_finetune_file, weights_only=False)
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']
    y_scaler = data['y_scaler']

    sequence_length = 365
    train_dataset = SequenceDataset(X_train, Y_train, sequence_length)
    test_dataset = SequenceDataset(X_test, Y_test, sequence_length)

**Model setup:**

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

**Training and testing:**

    model.train_model(loss_fun=loss_function, LR=learning_rate, step_size=step_size, gamma=gamma, maxepoch=max_epoch)
    model.test()

**Loss function desgin**

    from customize_loss import CarbonFluxLossCompiler

    script_config = {
        'parameters': {
            ...
            },

        'variables': {
            ...
            },
        
        'loss_fomula': {
            ...
            }
        }
    # Create the compiler
    compiler = CarbonFluxLossCompiler(script_config)

    # Create the loss function class
    CarbonFluxLoss = compiler.generate_class()
    loss_fn = CarbonFluxLoss()

**Model structure desgin**

    from customize_module import FlexibleModelCompiler

    script_config = {
        'class_name': 'my_KGML',
        'base_class': 'TimeSeriesModel',
        'init_params': {
            ...
            },

        'layers': {
            ...
            },
        'forward': {
            ...
            }
        }

    # Create the compiler
    compiler = FlexibleModelCompiler(script_config)

    # Create the model
    myKGML = Compiler.generate_model()

Details and example can be found in [unified_model_processing.ipynb](unified_model_processing.ipynb).


# Benchmark dataset
We integrated data from two prior studies to support the functional pipeline of PyKGML. These studies represent pioneering work in agricultural knowledge-guided machine learning (KGML), advancing the simulation of agroecosystem dynamics related to greenhouse gas (GHG) fluxes:

[Study 1](https://doi.org/10.5194/gmd-15-2839-2022): Liu et al. (2022). KGML-ag: a modeling framework of knowledge-guided machine learning to simulate agroecosystems: a case study of estimating N2O emission using data from mesocosm experiments, Geosci. Model Dev., 15, 2839–2858.

[Study 2](https://www.nature.com/articles/s41467-023-43860-5): Liu et al. (2024). Knowledge-guided machine learning can improve carbon cycle quantification in agroecosystems. Nature communications, 15(1), 357.

Study 1 developed KGML models to predict N<sub>2</sub>O fluxes using *ecosys* synthetic data as the pretraining dataset and chamber observations in a mesocosm environment as the fine-tuning dataset. Study 2 developed KGML models for predicting and partitioning CO<sub>2</sub> fluxes (Ra, Rh), using synthetic data in the pretraining step and flux tower observations for fine-tuning. [***ecosys***](https://github.com/jinyun1tang/ECOSYS) is a process-based model that incoporates comprehensive ecosystem biogeochemical and biogeophysical processes into its modeling design. To differentiate KGML models from the two studies, we refer to the selected model of study 1 as KGMLag-CO2 and that of study 2 as KGMLag-N2O.

Two datasets were harmonized using the CO<sub>2</sub> flux dataset from study 2 and and the N<sub>2</sub>O flux dataset from study 1 to demonstrate the use of PyKGML:
1. **CO<sub>2</sub> dataset:**
  * co2_pretrain_data:  
    - 100 samples (100 sites).
    - Each sample is a 6570 daily sequence over 18 years (2001-2018). 
    - 19 input_features and 3 output_features.    
    - Data split: the first 16 years for training, and the last two years for testing.

    Input features (19):
    - Meterological (7): solar radiation (RADN), max air T (TMAX_AIR), (max-min) air T (TDIF_AIR), max air humidity (HMAX_AIR), (max-min) air humidity (HDIF_AIR), wind speed (WIND), precipitation (PRECN).
    - Soil properties (9): bulk density (TBKDS), sand content (TSAND), silt content (TSILT), field capacity (TFC), wilting point (TWP), saturate hydraulic conductivity (TKSat), soil organic carbon concetration (TSOC), pH (TPH), cation exchange capacity (TCEC)
    - Other (3): year (Year), crop type (Crop_Type), gross primary productivity (GPP)

    Output features (3):
    - Autotrophic respiration (Ra), heterotrophic respiration (Rh), net ecosystem exchange (NEE). 
      
  * co2_finetune_data:  
    - One sample (11 sites were concatenated into one sequence due to their varied sequence lengths).
    - A Daily sequence of total 124 site-years (45260 in length).
    - 19 input_features and 2 output_features.
    - Data split: the last two years of each site were combined as the testing data, and the rest were included in the training data.  

    Input features (19):
    - The same as co2_pretrain_data.  
    
    Output features (2):
    - Ecosystem respiration (Reco, Reco = Ra + Rh), net ecosystem exchange (NEE). 

2. **N<sub>2</sub>O dataset:** 
  * n2o_pretrain_data:
    - 1980 simulations at 99 counties x 20 N-fertilizer rates in the 3I states (Illinois, Iowa, Indiana); synthetic data generated by ecosys.
    - Daily sequences over 18 years (2001-2018).
    - Data split: the first 16 years for training, and the last two years for testing.

    Input variables (16):  
    - Meterological (7): solar radiation (RADN), max air T (TMAX_AIR), min air T (TMIN_AIR), max air humidity (HMAX_AIR), min air humidity (HMIN_AIR), wind speed (WIND), precipitation (PRECN).
    - Soil properties (6): bulk density (TBKDS), sand content (TSAND), silt content (TSILT), pH (TPH), cation exchange capacity (TCEC), soil organic carbon concetration (TSOC)
    - Management (3): N-fertilizer rate (FERTZR_N), planting day of year (PDOY), crop type (PLANTT).

    Output variables (3):
    - N<sub>2</sub>O fluxes (N2O_FLUX), soil CO<sub>2</sub> fluxes (CO2_FLUX), soil water content at 10 cm (WTR_3), soil ammonium concentration at 10 cm (NH4_3), soil nitrate concentration at 10 cm (NO3_3).

  * n2o_finetune_augment_data:
    - Observations of 6 chambers in a mesocosm environment.
    - Daily sequences of 122 days x 3 years (2016-2018).
    - 1000 augmentations from hourly data at each chamber (6000 x 122 x 3 in total length).
    - Data split: 5 chambers as the training data, and the other one as the testing data.

    Input variables (16):  
    - Meterological (7): solar radiation (RADN), max air T (TMAX_AIR), min air T (TMIN_AIR), max air humidity (HMAX_AIR), min air humidity (HMIN_AIR), wind speed (WIND), precipitation (PRECN).
    - Soil properties (6): bulk density (TBKDS), sand content (TSAND), silt content (TSILT), pH (TPH), cation exchange capacity (TCEC), soil organic carbon concetration (TSOC)
    - Management (3): N-fertilizer rate (FERTZR_N), planting day of year (PDOY), crop type (PLANTT).

    Output variables (3):
    - N2O fluxes (N2O_FLUX), soil CO2 fluxes (CO2_FLUX), soil water content at 10 cm (WTR_3), soil ammonium concentration at 10 cm (NH4_3), soil nitrate concentration at 10 cm (NO3_3).

# PyKGML development
In PyKGML, we functionize several strategies for incoporating domain knowledge into the development of a KGML model a user-friendly way. Those strategies were explored and summarized in the two references ([Liu et al. 2022](https://doi.org/10.5194/gmd-15-2839-2022), [2024](https://www.nature.com/articles/s41467-023-43860-5)):
1. Pre-training and fine-tuning using benchmark datasets.
2. Knowledge-guided loss function customization according to physical or chemical laws in ecosystems, such as mass balance.
3. Hierarchical architecture design according to causal relationships.


PyKGML have realized loss function customization and model architecture design in a way to convert user's idea from an intuitive configuration script to a function or model using the loss function compiler or architecture compiler. Refer to [unified_model_processing.ipynb](unified_model_processing_v2.ipynb) for using examples. 

Models of KGMLag-CO2 and KGMLag-N2O were added to the model gallery of PyKGML so users can adopt these previously tested architectures for training or fine-tuning. Please note that the KGMLag-CO2 and KGMLag-N2O models in PyKGML only include the final deployable architectures of the original models, and do not include the strategies involved in pretraining and fine-tuning steps to improve the model performances. Instead, we generalize the process of model pre-training and fine-tuning for all models included in the gallery.

**Model gallery**:
  * Attention
  * GRUSeq2Seq
  * LSTMSeq2Seq
  * GRUSeq2SeqWithAttention
  * 1dCNN
  * TimeSeriesTransformer
  * N2OGRU_KGML (This is the model architecture of KGMLag-N2O from [Liu et al., 2022](https://doi.org/10.5194/gmd-15-2839-2022))
  * RecoGRU_KGML (This is the model architecture of KGMLag-CO2 from [Liu et al., 2024](https://www.nature.com/articles/s41467-023-43860-5)) 


# Acknowledgement
Funding sources for this research includes:  
1. This research is part of [AI-LEAF: AI Institute for Land, Economy, Agriculture & Forestry](https://cse.umn.edu/aileaf) and is supported by USDA National Institute of Food and Agriculture (NIFA) and the National Science Foundation (NSF) National AI Research Institutes Competitive Award No. 2023-67021-39829.
2. National Science Foundation: Information and Intelligent Systems (award No. 2313174).
3. The Forever Green Initiative of the University of Minnesota, using funds from the State of Minnesota Clean Water Fund provided by the Minnesota Department of Agriculture.  
4. National Science Foundation: Signal in the Soil program (award No. 2034385).

# Contact
Please contact the corresponding author Dr. Licheng Liu (lichengl@umn.edu) to provide your feedback.
