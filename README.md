# PyKGML development
PyKGML is a Python libaray to facilitate the development of kmowledge-guided machine learning (KGML) models for agricultural research.

# original studies
This repo incoporates two previous studies to demonstrate PyKGML:
study 1: Liu et al (2022). KGML-ag: a modeling framework of knowledge-guided machine learning to simulate agroecosystems: a case study of estimating N2O emission using data from mesocosm experiments, Geosci. Model Dev., 15, 2839–2858. https://doi.org/10.5194/gmd-15-2839-2022
study 2: Liu et al. (2024). Knowledge-guided machine learning can improve carbon cycle quantification in agroecosystems. Nature communications, 15(1), 357. https://www.nature.com/articles/s41467-023-43860-5

Study 1 pretrained models to predict N2O fluxes using ecosys synthetic data and fine tuned them using mesocosm data. Study 2 developed KGML for predicting and partitioning CO2 fluxes (Ra, Rh, NEE), using synthetic data in the pretraining step and flux tower observations for fine-tuning. To differentiate KGML models from the two studies, we call the model of study 1 KGMLag-CO2 and that of study 2 KGMLag-N2O.

# Files:
- Original code\n
  • KGMLag_CO2_pretrain_baseline.py and KGMLag_CO2_finetune_baseline.py are the original code from study 1. 
  • KGMLag_N2O_pretrain_baseline.ipynb and KGMLag_N2O_finetune_baseline.ipynb are the original code from study 2.
  
- Modularized code
  • KGMLag_CO2_pretrain_modularized.py are the modularized version of KGMLag_CO2_pretrain_baseline.py.
  • KGMLag_CO2_finetune_modularized.py are the modularized version of KGMLag_CO2_finetune_baseline.py.
  • KGMLag_N2O_modularized.py are the modularized version combining KGMLag_N2O_pretrain_baseline.py and KGMLag_N2O_finetune_baseline.py.
  • test_KGMLag_CO2_pretrain.ipynb is a testing call on KGMLag_CO2_pretrain_modularized.py
  • test_KGMLag_CO2_finetune.ipynb is a testing call on KGMLag_CO2_finetune_modularized.py
  
- packaged code
  • dataset.py is used to prepare demo datasets used for pretraining and fine-tuning: Step2_DataSet is the CO2 synthetic dataset for pretraining, and Step5_DataSet is the CO2 flux tower dataset for fine- 
  tuning. N2O_synthetic_dataset and N2O_mesocosm_dataset are the N2O pretraining and fine-tuning datasets, respectively.
  • kgml_lib.py contains utility functions such as for normalization and computing R2.
  • time_series_models.py assembles all the modularized code, and merges the processes of data spliting, model training and testing.
  • pretrain_CO2.ipynb and pretrain_N2O.ipynb are run files to showcase the use of time_series_models.py.

