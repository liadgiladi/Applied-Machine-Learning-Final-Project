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

1. Notebook - Applied-Machine-Learning-Final-Project-300822954-31143407.ipynb: 

   Note: please use colab.
   
   1.	Download notebook Applied-Machine-Learning-Final-Project-300822954-31143407.ipynb
   2.	Download classification_datasets-20200531T065549Z-001.zip
   3.	Download results/experiments_results.csv
   4.	Download ClassificationAllMetaFeatures.csv
   5. Open Applied-Machine-Learning-Final-Project-300822954-31143407.ipynb notebook in colab
   6.	The following files are needed to be uploaded to the notebook under '/content' dir (which is the default):
        * classification_datasets-20200531T065549Z-001.zip
        * results/experiments_results.csv (this file is not required if run_nested_cross_validation flag is set to 'True')
        * ClassificationAllMetaFeatures.csv
   7. Run all cells
  
   There are two options to run the notebook:
     * Skip the nested-cross-validation section and run only the processing results section + meta learning section. Default option. 
     * Run the whole exercise (nested-cross-validation section + processing the results section + meta learning section). In order to that, please set in the settings cell (#2         cell) ‘run_nested_cross_validation = True’. Please note that this might take a while.
   
2. Directly: 
   run flow.py file.
   Note: this option is less favorable and allow to run the experiment only, does not support other sections of the project, such as statistical hypothesis test, meta-learning-          model, graphs, etc.

