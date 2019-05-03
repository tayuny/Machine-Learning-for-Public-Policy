import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
from sklearn import impute
from sklearn.feature_selection import SelectKBest

STRATEGY = ["mean", "median", "most_frequent", "constant"]

# Missing Data
#####################################################################################

# Explore Data
def summarize_missing_values(df):
    '''
    This function is used to summarize the missing values
    in specific columns in a given dataframe
    Inputs:
    '''
    missing_in_column_dict = {}
    for column in list(df.columns):
        column_null = df[df[column].isnull()]
        missing_in_column_dict[column] = (column_null.shape[0], df.shape[0])
    
    return missing_in_column_dict


def summarize_data_frame(df, corr=False):
    '''
    This function is used to summarize the summary statistics
    of df, it can also be used to generate correlation matrix
    Inputs:
        df: dataframe
        corr: (boolean) return correlation matrix if True
    Returns: summary statistics (dataframe), correlation matrix (dataframe)
    '''
    if corr:
        return df.describe(), df.corr()

    return df.describe()


####################################################################
# Source: Plotting a diagonal correlation matrix, Seaborn Gallery
# Author: Michael Waskom
# Date: 2012-2018
####################################################################
def plot_corr_metrix(corr_df):
    '''
    This function is used to plotted correlation matrix with different color
    representing different intervals of value.
    Inputs:
        corr_df: dataframe of correlation matrix
    '''
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0, \
             square=True, linewidths=.5, cbar_kws={"shrink": .5})


def fill_na(df):
    '''
    This function is used to fill the mean of every columns in the missing cell
    Inputs:
        df: dataframe
    Returns: dataframe with missing values filled
    '''
    mean_dict = {}
    for column in list(df.columns):
        mean_dict[column] = np.mean(df[column])
    
    df = df.fillna(value=mean_dict)
    
    return df

def fill_na_freq(df):
    '''
    The function is used to fill in the missing value with value which appears most
    frequently in the column.
    Inputs:
        df: dataframe
    Returns: dataframe with missing values filled
    '''
    freq_dict = {}
    for column in df.columns:
        most_freq = df.groupby(column).size().sort_values(ascending=False).index[0]
        freq_dict[column] = most_freq
    df = df.fillna(value=freq_dict)

    return df

def univariate_imputation(df, missing_values=np.nan, strategy="mean"):
    '''
    This function is used to provide univariate imputation method to fill in the
    missing values of a given DataFrame
    Inputs:
        df: DataFrame
        missing_values: indicator of the missing value in the data
        strategy: the initial strategy used to impute the value
    Returns: dataframe with missing values filled
    '''
    imp_model = impute.SimpleImputer(missing_values=missing_values, \
                              strategy=strategy, fill_value=None, \
                              verbose=0, copy=True)
    columns = list(df.columns)
    df = imp_model.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)

    return df


# Currentlt only available for scikit-learn version 0.21.dev0
def multiple_imputation(df, add_indicator=False, estimator=None,\
         imputation_order='ascending', initial_strategy='mean',\
         max_iter=10, missing_values=np.nan,\
         n_nearest_features=None, random_state=None,\
         sample_posterior=False):
    '''
    This function is used to provide multivariate imputation method to fill in the
    missing values of a given DataFrame
    Inputs:
        df: DataFrame
        missing_values: indicator of the missing value in the data
        initial_strategy: the initial strategy used to impute the value
        n_nearest_features: select n features used in the multivariate method which
                            have n highest correlation with the column contains missing
                            values.
    Returns: dataframe with missing values filled
    '''
    imp_model = impute.IterativeImputer(add_indicator=add_indicator, \
                    estimator=estimator, imputation_order='ascending',\
                    initial_strategy='mean', max_iter=10, \
                    missing_values=np.nan, n_nearest_features=None,\
                    random_state=None, sample_posterior=False)
    
    columns = list(df.columns)
    df = imp_model.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    
    return df


# Feature Selection
##########################################################################################

def select_K_best_features(df, features, classifier, k=10):
    '''
    The function is used to select K features which contain K highest
    variance
    Inputs:
        df: dataframe
        features: features used in selection
        classifier: classifier should always be remained
        k: number of highest k variance
    Return: 
        f_final: the dataframe with the features selected
        f_score: Series of features and their corresponding scores
    '''
    f_fit = SelectKBest(k=k).fit(df[features], df[classifier])
    f_final = f_fit.transform(df[features])
    
    f_score = f_fit.scores_
    f_score = pd.Series(f_score, index=features).sort_values(ascending=False)[:k]

    return f_final, f_score


def category_to_binary(df, column):
    '''
    This function is used to transform each of the category in the categorical
    variable to a binary variable
        Inputs: dataframe
        column: given categorical variable
    Return: updated dataframe
    '''
    for category in df[column].unique():
        df[column + "==" + str(category)] = 0
        df.loc[df[column] == category, column + "==" + str(category)] = 1

    return df


def combination_indexer(var_len, index_list):
    '''
    This function is used to provide binary index for the list of variables
    which will be used in get_var_combinations function
    Inputs:
        var_len: the length of the variable list
        index_list: list of the indices
    Returns: a list with all possible binary combinations (list of list)
    '''
    if len(index_list) == var_len:
        return [index_list]
    
    else:
        total_list = []
        for i in range(2):
            new_list = index_list + [i]
            total_list += combination_indexer(var_len, new_list)
    
    return total_list