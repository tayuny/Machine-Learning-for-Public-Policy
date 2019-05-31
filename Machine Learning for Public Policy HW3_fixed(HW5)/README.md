## Machine Learning for Public Policy fixed HW3 (HW5)
This purpose of this assignment is to improve machine learning pipeline in HW3.

### The folder contains files with .py, .ipynb, pdf and pickle data
#### py files
* data_util.py: functions for data type transformation and summary statistics
* train_test_split.py: functions for train_test splitting
* imputation.py: functions dealing with missing data
* feature_generation.py: functions dealing with feature generations
* evaluation.py: functions of evaluation methods
* clf_define.py: functions implementing classifier definition and implement training, testing, and evaluations over time
* unsupervised.py: functions of unsupervised learning models and visualization

#### ipynb files: The first part of three of the ipynb files are the same, they all contains the main data pre-processing step before the model is trained
* HW3_fixed_simple.ipynb: This file contains all the classifiers with simple parameters, it is used as a proof that the pipeline is able to be implemented and the plotting is performing
* HW3_fixed-sub_small.ipynb: This file contains the high priority classifiers (decisiion tree, random forest, logistics) with the analysis of the optimal classifier and performance over time
* HW3_fixed-sub_small_others.ipynb: This file is used to generate the full performance matrix (save as pickle files) for general evaluations for all models over time.

#### pickle data:
* cross_val_performance_small_final: pickle files contains the full performance matrix for classifiers except SVM and KNN
* cross_val_performance_other_final: pickle files contains the performance matrix for classifiers SVM and KNN

### Dependency:
* python 3.7
* pandas 0.23.4
* numpy 1.14.5
* seaborn 0.9.0
* matplotlib 2.2.2
* graphviz 0.10.1
* scikit-learn 0.20.3, 0.21.dev0 (if IterativeImputer is used)

### Reference:
* Reference: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
