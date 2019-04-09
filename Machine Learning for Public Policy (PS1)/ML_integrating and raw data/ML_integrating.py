import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import geopy
import ML_util

# Import Crime Data
crime17 = pd.read_csv(r"Crimes2017.csv")
crime18 = pd.read_csv(r"Crimes2018.csv")

def clean_dataset(df):
    '''
    This function is used to select the data with variables
    of community and locations
    Input: crime dataframe
    Returns: updated crime dataframe
    '''
    df = df[df["Community Area"].notnull()]
    df["Community Area"] = df["Community Area"].astype(int)
    df = df[df["Location"].notnull()]

    return df

# Clean crime data
crime17_cleaned = clean_dataset(crime17)
crime17_cleaned.to_csv("crime17_cleaned")
crime18_cleaned = clean_dataset(crime18)
crime18_cleaned.to_csv("crime18_cleaned")

# Import American Community Survey Data for Cook County
population = pd.read_csv(r"ACS_17_5YR_B01003_with_ann.csv")
median_income = pd.read_csv(r"ACS_17_5YR_B19013_with_ann.csv")
race = pd.read_csv(r"ACS_17_5YR_B02001_with_ann.csv")
gini = pd.read_csv(r"ACS_17_5YR_B19083_with_ann.csv")
employment = pd.read_csv(r"ACS_17_5YR_S2301_with_ann.csv")
poverty = pd.read_csv(r"ACS_17_5YR_S1702_with_ann.csv")

# Import American Community Survey Data for Great Chicago Area
c_population = pd.read_csv(r"\ACS_metro\ACS_17_5YR_B01003_with_ann.csv", \
                                          encoding="latin_1")
c_median_income = pd.read_csv(r"\ACS_metro\ACS_17_5YR_B19013_with_ann.csv", \
                                          encoding="latin_1")
c_race = pd.read_csv(r"\ACS_metro\ACS_17_5YR_B02001_with_ann.csv", \
                                              encoding="latin_1")
c_gini = pd.read_csv(r"\ACS_metro\ACS_17_5YR_B19083_with_ann.csv", \
                                              encoding="latin_1")
c_employment = pd.read_csv(r"\ACS_metro\ACS_17_5YR_S2301_with_ann.csv", \
                                                   encoding="latin_1")
c_poverty = pd.read_csv(r"\ACS_metro\ACS_17_5YR_S1702_with_ann.csv", \
                                                encoding="latin_1")

def select_ACS_df(population, median_income, race, gini, employment, poverty):
    '''
    This function is used to merge datas for American Community Survey
    Inputs: dataframe with population, median income, race information
            gini index, employment status and poverty rate
    Rerturns: intergrated dataframe
    '''
    population = population.iloc[1:][[\
                 "GEO.id2", "GEO.display-label", "HD01_VD01"]].rename(\
                 columns={"GEO.id2":"zipcode", 
                          "GEO.display-label":"name", 
                          "HD01_VD01":"population"})
    median_income = median_income.iloc[1:][[\
                    "GEO.id2","HD01_VD01"]].rename(\
                    columns={"GEO.id2":"zipcode", \
                             "HD01_VD01":"median_income"})
    race = race.iloc[1:][["GEO.id2", "HD01_VD02", "HD01_VD03"]].rename(\
    	   columns={"GEO.id2":"zipcode", "HD01_VD02":"White", \
    	            "HD01_VD03":"African American"})
    gini = gini.iloc[1:][["GEO.id2", "HD01_VD01"]].rename(\
    	   columns={"GEO.id2":"zipcode", "HD01_VD01":"gini_index"})
    employment = employment.iloc[1:][[\
                 "GEO.id2", "HC02_EST_VC01", "HC04_EST_VC46"]].rename(\
                 columns={"GEO.id2":"zipcode", \
                          "HC02_EST_VC01":"labor_participation", \
                          "HC04_EST_VC46":"unemployment"})
    poverty = poverty.iloc[1:][["GEO.id2", "HC02_EST_VC01"]].rename(\
    	      columns={"GEO.id2":"zipcode", "HC02_EST_VC01":"poverty_rate"})

    ACS_df = pd.merge(population, median_income, left_on="zipcode", \
                                                 right_on="zipcode")
    ACS_df = pd.merge(ACS_df, race, left_on="zipcode", right_on="zipcode")
    ACS_df = pd.merge(ACS_df, gini, left_on="zipcode", right_on="zipcode")
    ACS_df = pd.merge(ACS_df, employment, left_on="zipcode", \
                                          right_on="zipcode")
    ACS_df = pd.merge(ACS_df, poverty, left_on="zipcode", right_on="zipcode")

    return ACS_df

# Intergrate American Community Survey Data
ACS_df = select_ACS_df(population, median_income, race, gini, \
                                         employment, poverty)
ACS_df_metro = select_ACS_df(c_population, c_median_income, \
                   c_race, c_gini, c_employment, c_poverty)
Chicago_metro = ACS_df_metro[ACS_df_metro["name"]==\
                "Chicago-Naperville-Elgin, IL-IN-WI Metro Area"]

ACS_df.to_csv("ACS_df.csv")
ACS_df_metro.to_csv("ACS_df_metro.csv")