# Spliting data to train and test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# General
def train_test_general(df, features, classifier, test_size=0.1, seed=None):
    '''
    This function provide the most general method to split the data to
    training and testing subset.
    Inputs:
        df: dataframe
        features: features selected
        classifier: classifier used as the indicator of output
        test_size: the proportion of the testing subset
    Returns: X_train, y_train, X_test, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(df[features], \
                                                        df[classifier], \
                                                        test_size=test_size,\
                                                        random_state=seed)
    return X_train, y_train, X_test, y_test


# Cross Sections
def train_test_KFold(df, features, classifier, split=4, shuffle=False, random_state=None):
    '''
    This function is used to provide KFold split to training and testing subsets
    Inputs:
        df: dataframe
        features: features selected
        classifier: classifier used as the indicator of output
        split: the number of ways to seperate training and testing subsets
    Returns: dictionary of [X_train, y_train, X_test, y_test]
    '''
    data_dict = {}
    i = 0
    dfs = KFold(n_splits=split, shuffle=shuffle, random_state=random_state)

    for train, test in dfs.split(df):
        X_train = df.iloc[train][features]
        X_test = df.iloc[test][features]
        y_train = df.iloc[train][classifier]
        y_test = df.iloc[test][classifier]
        data_dict[i] = [X_train, y_train, X_test, y_test]
        i += 1

    return data_dict


# Temporal
def temporal_split(df, time_index, features, classifier, time_interval):
    '''
    This function is used to provide temporal split to training and testing subsets,
    data in the time_interval before the testing data will be considered as the
    corresponding training data
    Inputs:
        df: dataframe
        time_index: the feature used as the time variable
        features: features selected
        classifier: classifier used as the indicator of output
        time_interval: the length of time for the trainind subset
    Returns: dictionary of [X_train, y_train, X_test, y_test]
    '''
    df = df.sort_values(by=[time_index])
    time_order = sorted(list(df[time_index].unique()))
    time_order = time_order[:-time_interval]
    # Test index is located in the last location of lists in index_list
    index_dict = {}

    for time in time_order:
        train_indices = list(np.arange(time, time + time_interval))
        train = df[df[time_index].isin(train_indices)]
        test = df[df[time_index] == time + time_interval]
        sub_list = [train[features], train[classifier], test[features], test[classifier]]
        index_dict[time] = sub_list
    
    return index_dict


def rolling_window_split(df, time_index, features, classifier):
    '''
    This function is used to provide temportal split to training and testing subsets with
    rolling window strategy
    Inputs:
        df: dataframe
        time_index: the feature used as the time variable
        features: features selected
        classifier: classifier used as the indicator of output
    Returns: dictionary of [X_train, y_train, X_test, y_test]
    '''
    index_dict = {}
    for time in sorted(list(df[time_index].unique()))[1:]:
        test = df[df[time_index] == time]
        train = df[df[time_index] < time]
        sub_list = [train[features], train[classifier], test[features], test[classifier]]
        index_dict[time] = sub_list

    return index_dict


# Reset Balance
def reset_balance(df, classifier, sample_times, cls0_size=1, cls1_size=1):
    '''
    This function is used to sample the majority in the unbalance classification
    Inputs:
        df: dataframe
        classifier: name of classifier in the decision tree
        sample_times: time used for resampling
        cls0_size: relative size of classification 0
        cls1_size: relative size of classification 1
    Returns: list of balanced dataframe
    '''
    df_list = []
    
    if cls0_size >= cls1_size:
        base_size = df[df[classifier] == 1].shape[0]
        sample_size = base_size * cls0_size
        base_df = df[df[classifier] == 1]
        sample_df = df[df[classifier] == 0]

    else:
        base_size = df[df[classifier] == 0].shape[0]
        sample_size = base_size * cls1_size
        base_df = df[df[classifier] == 0]
        sample_df = df[df[classifier] == 1]

    for i in range(sample_times):
        sub_sample_df = sample_df.sample(sample_size, axis=0)
        whole_sub_df = pd.concat([sub_sample_df, base_df], join="inner")
        df_list.append(whole_sub_df)

    return df_list