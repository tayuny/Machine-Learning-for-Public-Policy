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


def fill_unknown(train_df, test_df, categorical_columns):
    '''
    The function is used to fill in the missing value with "unknown".
    Inputs:
        df: dataframe
    Returns: dataframe with missing values filled
    '''
    for column in categorical_columns:
        train_df = train_df.fillna(value="unknown")
        test_df = test_df.fillna(value="unknown")

    return train_df, test_df


def fill_na_freq(train_df, test_df):
    '''
    The function is used to fill in the missing value with value which appears most
    frequently in the column.
    Inputs:
        df: dataframe
    Returns: dataframe with missing values filled
    '''
    freq_dict = {}
    for column in train_df.columns:
        most_freq = df.groupby(column).size().sort_values(ascending=False).index[0]
        freq_dict[column] = most_freq
    train_df = train_df.fillna(value=freq_dict)
    test_df = test_df.fillna(value=freq_dict)

    return train_df, test_df


def fill_na_mean(train_df, test_df, continuous_columns):
    '''
    The function is used to fill in the missing value with mean of the training data
    Inputs:
        df: dataframe
    Returns: dataframe with missing values filled
    '''
    mean_dict = {}
    for column in continuous_columns:
        col_mean = train_df[column].mean()
        mean_dict[column] = col_mean
    train_df = train_df.fillna(value=mean_dict)
    test_df = test_df.fillna(value=mean_dict)

    return train_df, test_df


def univariate_imputation(train_df, test_df, missing_values=np.nan, strategy="mean"):
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
    train_df = imp_model.fit_transform(train_df)
    test_df = imp_model.transform(test_df)

    train_df = pd.DataFrame(train_df, columns=columns)
    test_df = pd.DataFrame(test_df, columns=columns)

    return train_df, test_df


# Currentlt only available for scikit-learn version 0.21.dev0
def multiple_imputation(train_df, test_df, add_indicator=False, estimator=None,\
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
    train_df = imp_model.fit_transform(train_df)
    test_df = imp_model.transform(test_df)

    train_df = pd.DataFrame(train_df, columns=columns)
    test_df = pd.DataFrame(test_df, columns=columns)
    
    return train_df, test_df


class community_mean_imputer:
    '''
    The class is designed to implement imputation with regional mean in given time
    '''
    def __init__(self):
        self.trained_imp = {}
        self.filled_category = ""


    def filled_categorical(self, train_df, test_df, categorical_columns):
        '''
        The function is used to fill in unknown for the missing values in categorical
        variables
        Inputs:
            train_df: training dataframe
            test_df: testing dataframe
            categorical_columns: list of columns with categorical variables
        Returns: train_df, test_df
        '''
        self.filled_category = categorical_columns
        for column in categorical_columns:
            train_df[column].fillna("unknown")

        for test_col in categorical_columns:
            test_df[test_col].fillna("unknown")

        return train_df, test_df


    def train_regional_mean(self, df, loc_column, time_column):
        '''
        The function is used to trian the model of imputation
        Inputs:
            df: dataframe
            loc_column: column represents the geographical unit
            time_column: column represents time unit
        Returns: the imputed trained dataframe
        '''

        used_col_list = list(df.columns)
        for i in set([loc_column, time_column] + self.filled_category):
            used_col_list.remove(i)

        for col in used_col_list:
            print(col)

            for loc in list(df[loc_column].unique()):
                for year in list(df[time_column].unique()):
                    condition = (df[loc_column] == loc) & (df[time_column] == year)
                    df.loc[(condition & df[col].isnull()), col] = df.loc[condition, col].mean()

                    if (loc, year) not in self.trained_imp:
                        self.trained_imp[(loc, year)] = {}
                    self.trained_imp[(loc, year)][col] = df.loc[condition, col].mean()

        #updated_missing_dict != summarize_missing_values(df)
        #for column, values in updated_missing_dict.items():
        #    if values[0] == 0:
        #        df.loc[df[column].isnull(), column] = df[column].mean()
        #        self.trained_imp[(loc, year)][col] = df.loc[column].mean()

 
        return df

    def transform_test(self, test_df, loc_column, time_column):
        '''
        This model is used to test the imputation model trained by the regional mean imputer
        Inputs:
            test_df: the testing dataframe
            loc_column: column represents the geographical unit
            time_column: column represents time unit
        Returns: imputed test dataframe
        '''
        used_col_list = list(test_df.columns)
        for i in set([loc_column, time_column] + self.filled_category):
            used_col_list.remove(i)

        for column in used_col_list:
            for loc in list(test_df[loc_column].unique()):
                for year in list(test_df[time_column].unique()):
                    condition = ((test_df[column].isnull()) & (test_df[loc_column] == loc) & (
                                test_df[time_column] == year))
                    test_df.loc[condition, column] = self.trained_imp[(loc, year - 2)][column]

        return test_df