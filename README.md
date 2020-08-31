# Applied-Machine-Learning-Final-Project

## General Information:
Team Name: Liad & Yuval

IDs:

ID #1: 300822954

ID #2: 311434047

## This repository contains the following files:

  1. Applied-Machine-Learning-Final-Project-300822954-31143407.ipynb - which is a notebook contains all relevant code. Allow to run the experiment and contains other sections of      the project such as statistical hypothesis test, meta-learning-model, graphs, etc. **Was tested on Colab**. 
  
  2. flow.py - experiement flow controller + hyper-parameter search grid.
  
  3. nested_cv.py - nested cross-validation infrastructure.
  
  4. utils.py
  
  5. rotboost.py - RotBoost implementation
  
  6. rotation_forest.py - Rotation-Forest implementation
  
  7. results/experiments_results.csv - our experiments results as depicted in the file 'Final-Project-Report-300822954-31143407.docx'
  
  8. classification_datasets-20200531T065549Z-001.zip - holds all csv datasets zipped
  
  9. /classification_datasets - holds all csv datasets
  
  10. ClassificationAllMetaFeatures.csv - meta learning model's meta features
  
  11. dataset-metadata.csv - holds datasets metadata such as binary/multiclass type, number of attributes etc. 
  
  12. results/feature_importance_weight.png - meta learning model feature importance type 'weight'
  
  13. results/feature_importance_gain.png - meta learning model feature importance type 'gain'
  
  14. results/feature_importance_cover.png - meta learning model feature importance type 'cover'
  
  15. results/shap_summary_plot.png
  
  16. results/shap_training_set_prediction.png
  
  17. results/hyperparameters-search-space.png
  
  18. Applied-Machine-Learning-Final-Project-300822954-31143407.docx

# How to run:

1. Notebook: 

   Note: please use colab.
   
   The following files are needed to be uploaded under '/content' dir (which is the default):
   1. classification_datasets-20200531T065549Z-001.zip
   2. results/experiments_results.csv (this file is not required if run_nested_cross_validation flag is set to 'True')
   3. ClassificationAllMetaFeatures.csv
   
   In order to run the experiment you should set 'run_nested_cross_validation = True' under 'settings' cell in the notebook. Please note that this might take a while. 
   
2. Directly: 
   run flow.py file.
   Note: this option is less favorable and allow to run the experiment only, does not support other sections of the project, such as statistical hypothesis test, meta-learning-          model, graphs, etc.

