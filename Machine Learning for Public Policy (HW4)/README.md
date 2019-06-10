## Machine Learning for Public Policy HW4

The purpose of this assignment is to develop pipeline for unsupervised learning.

### The folder contains files with .py, .ipynb, and pdf file

#### pdf file

* Machine Learning for Public Policy HW4 final_report.pdf: The final report for this analysis.

### py file

* unsupervised.py: functions of unsupervised learning models and visualization (main py file in this task)
* data_util.py: functions for data type transformation and summary statistics
* train_test_split.py: functions for train_test splitting
* imputation.py: functions dealing with missing data
* feature_generation.py: functions dealing with feature generations
* evaluation.py: functions of evaluation methods
* clf_define.py: functions implementing classifier definition and implement training, testing, and evaluations over time


### ipynb files: The first part of three of the ipynb files are the same, they all contains the main data pre-processing step before the model is trained
data_transformer.ipynb: This notebook contains the note to derive the cleaned dataset used for testing
unsupervised_testing.ipynb: This notebook displays the functions in the unsupervised pipeline and generate the result use in the analysis

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
