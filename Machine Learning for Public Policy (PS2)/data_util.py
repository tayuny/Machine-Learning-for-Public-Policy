import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters

COLUMN_ORDERED = ['PersonID', 'zipcode', 'age', 'RevolvingUtilizationOfUnsecuredLines', 
                  'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 
                  'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                  'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
                  'NumberOfDependents', 'SeriousDlqin2yrs']
# Read Data
def read_csv_data(path, dtype_dict={}):
    '''
    This function is used to read and process csv file.
    Input:
        path: path of csv_file
        dtype_dict: dictionary of datatype for columns in
               the dataframe
    Return: pandas dataframe
    '''
    df = pd.read_csv(path, dtype=dtype_dict)
    return df


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
        return df.describe(), df[COLUMN_ORDERED[2:]].corr()

    return df.describe()


def compare_means(df, column):
    '''
    This function is used to compare the means and other summary statistics of
    all variables for every categories in the given column
    Inputs:
        df: dataframe
        column: given column
    return: dictionary which includes summary statistics for every category
    '''
    sum_df_dict = {}
    for category in (df[column].unique()):
        sum_df_dict[column + "==" + str(category)] = df[df[column]==category].describe()
        
    return sum_df_dict


def depict_group(df, var1, var2, var3=False):
    '''
    The function is used to depict the scatter plot of two variables,
    if the third variables is provided, categories of the third variable
    will be plotted with different colars in the plot.
    Inputs:
        df: dataframe
        var1: variable in X axis
        var2: variable in Y axis
        var3: (optional) variable plotted with different colors with its
                         different categories
    '''
    if var3:
        sns.relplot(x=var1,y=var2, hue=var3, data=df)
    else:
        sns.relplot(x=var1,y=var2, data=df)


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
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


def get_pair_distribution(df, variables):
    '''
    This function is used to plot the graph of pair distributions of 
    given variables
    Inputs:
        df: dataframe
        variables: variables given to be plotted
    '''
    sns.pairplot(df, vars=variables)


def plot_outlier_of_variable(df, column):
    '''
    This function is used to provide the boxplot in search of distribution
    and outliers of the given column
    Inputs:
        df: dataframe
        column: column given to be plotted
    '''
    boxplot = sns.boxplot(x=df[column])
    histogram = df.hist(column=column)


def distribution_for_all_columns(df):
    '''
    This function is used to plot the graph of distributions of all variables
    in the dataframe
    Inputs:
        df: dataframe
    '''
    sns.boxplot(data=df, orient="h")


# Pre-process Data
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


# Generate features/predictors
def continue_to_category(df, column, interval_list, category_list):
    '''
    This function is used to transform continuous variables to categorical 
    variables
    Inputs:
        df: dataframe
        column: given column to be transformed
        interval_list: list of the interval of the categories
        category_list: list of names of corresponding categories
    Returns: updated dataframe
    '''
    df[column + "_category"] = pd.cut(df[column], bins=interval_list, \
                                     labels=category_list, \
                                     include_lowest=False, right=True)
    return df


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


def get_var_combinations(var_list):
    '''
    This function is used to provide all possible combinations of variables
    which will be used in the input of machine learning methods
    Inputs:
        var_list: a list of variables
    Returns: list of all possible combinations of variables in the given
             variable list (list of list)
    '''
    index_combination = combination_indexer(len(var_list), [])
    var_combinations = []
    
    for combination in index_combination:
        var_sublist = []
        for i, val in enumerate(combination):
            if val == 1:
                var_sublist.append(var_list[i])

        var_combinations.append(var_sublist)
    
    return var_combinations