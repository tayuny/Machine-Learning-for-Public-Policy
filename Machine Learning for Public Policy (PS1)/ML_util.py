import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import geopy

MAIN_CRIMES = ['CRIM SEXUAL ASSAULT', 'HOMICIDE', 'ROBBERY', 'ASSAULT',\
                   'BURGLARY', 'BATTERY','MOTOR VEHICLE THEFT', 'ARSON']

def gen_community_crime_type_df(df):
    '''
    The function is used to generate the dataframe of which presents the
    count of each categotries of severe offenses in each community in Chicago
    Inputs: crime dataframe
    Return: target crime dataframe
    '''
    df_type_com = df.groupby(("Community Area", "Primary Type")).size()
    df_type_com = df_type_com.unstack().fillna(value=0)
    df_type_com = df_type_com.astype(int)
    df_type_com = df_type_com[MAIN_CRIMES]

    return df_type_com

def depict_top_n_community(df, n, crime_type):
    '''
    This function is used to return the top n communities with the 
    highest occurances for a given crime type
    Inputs:
        df: crime dataframe
        n: top n communities (integer)
        crime_type: type of crime defined in the crime dataset (string)
    Returns: dataframe with information of occurances.
    '''
    df = df[df["Primary Type"] == crime_type]
    top_n = df.groupby("Community Area").size().sort_values(\
                                   ascending=False)[:n].index
    df = df[df["Community Area"].isin(top_n)]
    df_depict = df.groupby(("Community Area", "Primary Type"))\
                  .size().to_frame().rename(columns={0:"Count"})\
                  .sort_index(level=1).reset_index()

    return df_depict

def plot_top_n_community(df, n, crime_list):
    '''
    This function is used to plot the information of the top n communities 
    with highest occurances for given list of crime types
    Inputs:
        df: crime dataframe
        n: top n communities (integer)
        crime_list: list of crime types (list of string)
    Returns: catplot with information of occurances
    '''
    df = df[df["Primary Type"].isin(crime_list)]
    top_n = df.groupby("Community Area").size().sort_values(\
                                   ascending=False)[:n].index
    df = df[df["Community Area"].isin(top_n)]
    df_depict = df.groupby(("Community Area", "Primary Type"))\
                  .size().to_frame().rename(columns={0:"Count"}).\
                  sort_index(level=1).reset_index()
    plt = sns.catplot(x="Primary Type", y="Count", hue="Community Area", \
                data=df_depict, kind="bar", height=20, aspect=0.8, orient="v")
    return plt

def compare_crime_change_in_community(df1, df2, crime_type):
    '''
    This function is used to generate a table to compare crime data 
    in community level in two different year for a given crime_type
    Input: df1, df2: two Dataframe from 2 years
    Return: target Dataframe with the comparison
    '''
    df1 = df1[df1["Primary Type"] == crime_type]
    df1_com = df1.groupby("Community Area").size().to_frame().\
                                rename(columns={0:"Count17"})
    df1_com["Community Area"] = df1_com.index
    df2 = df2[df2["Primary Type"] == crime_type]
    df2_com = df2.groupby("Community Area").size().to_frame().\
                                rename(columns={0:"Count18"})
    df2_com["Community Area"] = df2_com.index
    df = pd.merge(df1_com, df2_com, left_on="Community Area", \
                                   right_on="Community Area")
    df = df[["Community Area", "Count17", "Count18"]]
    df["Difference"] = df["Count18"] - df["Count17"]
    hist = sns.distplot(df["Difference"])

    return df, hist

def sampling_crime_in_community(df, crime_type, community):
    '''
    This function is used to sample crime occurances in a given community 
    and get geographical information (zipcode) used to merge with 
    American Community Survey data
    Inputs:
        df: crime dataframe
        crime_type: type of crime (string)
        community: community number in Chicago (string)
    Returns: sub-dataframe with geographical information
    '''
    df = df[df["Primary Type"]==crime_type]
    df = df[df["Community Area"]==community]
    
    if df.shape[0] >= 10:
        df_sample = df.sample(10, axis=0)
    else:
        df_sample = df
        
    df_sample["cor_str"] = df_sample.apply\
                           (lambda x: str(x["Latitude"]) + "," + \
                            str(x["Longitude"]) \
                            if not x["Latitude"] is np.nan \
                            else np.nan, axis=1)
    df_sample["full_address"] = df_sample.apply\
                               (lambda x: str(geolocator.reverse(x["cor_str"]))\
                                if not x["cor_str"] == "nan,nan" \
                                else np.nan, axis=1)
    df_sample["zipcode"] = df_sample.apply(lambda x: \
                           re.search(("(Illinois,\s)([0-9]{5,6})"), \
                           x["full_address"]).group(2)\
                           if not (x["full_address"] is np.nan)\
                           else np.nan\
                           if not (re.search(("(Illinois,\s)([0-9]{5,6})"),\
                                             x["full_address"]) is np.nan)\
                           else np.nan, axis=1)
    return df_sample

def search_community_crime_char(crime_df, top_n_df, crime_type):
    '''
    The function is used to intergrate the dataframe which includes crime 
    and geographical information for n communities which have the highest 
    occurances of given crime type
    Inputs:
        crime_df: crime dataframe
        top_n_df: dataframe contains the occurances of crime in communities
        crime_type: type of crime (string)
    Returns: integrated dataframe
    '''
    com_list = list(top_n_df[top_n_df["Primary Type"] == \
                    crime_type]["Community Area"])

    final_columns = list(crime_df.columns) + ["cor_str"] + \
                          ["full_address"] + ["zipcode"]
    final_df = pd.DataFrame(columns=final_columns)
    for community in com_list:
        com_df = sampling_crime_in_community(crime_df, crime_type, community)
        final_df = pd.concat([final_df, com_df], join="inner")
    
    return final_df

def search_multiple_crime_type(crime_df, top_n, crime_types, ACS_df):
    '''
    The function is used to intergrate the dataframe which includes ACS, 
    crime and geographical information for n communities which have the 
    highest occurances of given crime types
    Inputs:
        crime_df: crime dataframe
        top_n_df: dataframe contains the occurances of crime in communities
        crime_types: type of crime (list of string)
        ACS_df: dataframe of American Community Survey
    Returns: dictionary which maps crime_type to an output dataframe
    '''
    crime_type_dict = {}
    for crime_type in crime_types:
        top_n_df = depict_top_n_community(crime_df, top_n, crime_type)
        crime_subdf = search_community_crime_char(crime_df, top_n_df, \
                                                            crime_type)
        crime_subdf = pd.merge(crime_subdf, ACS_df, how="left", \
                               left_on="zipcode", right_on="zipcode")
        crime_type_dict[crime_type] = crime_subdf
    
    return crime_type_dict

def compute_crime_change_in_type(df1, df2):
    '''
    This function is used to compute the differences in type of crime 
    in different years
    Input: dataframe from two different years
    Returns: dataframe with infomation of the comparison
    '''
    df1_s = df1.groupby("Primary Type").size().to_frame().\
                                        rename(columns={0:"Count17"})
    df2_s = df2.groupby("Primary Type").size().to_frame().\
                                        rename(columns={0:"Count18"})
    df = pd.merge(df1_s, df2_s, right_on="Primary Type", \
                                left_on="Primary Type")
    df["Difference_rate"] = (df["Count18"] - df["Count17"]) / df["Count17"]
    
    return df