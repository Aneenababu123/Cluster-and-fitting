# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:15:14 2023

@author: DELL
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#%matplotlib inline


def read_data(filename):
    '''
    This is a function called read_data that takes a filename as an input
    '''
    df = pd.read_excel(filename, skiprows = 3)
    return df

def stat_data(df, col, value, yr, a):
    
    """This function takes a pandas DataFrame df, a column name col, a value to filter on value
    """
    df3 = df.groupby(col, group_keys = True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    df3 = df3.transpose()
    df3 = df3.loc[:,a ]
    df3 = df3.dropna(axis = 1)
    return df3

def Expo(t, scale, growth):
    ''' The function calculates the value of an exponential function at a given time point, using the specified scaling and growth factors'''
    f = (scale * np.exp(growth * (t-1960)))
    return f

def func(x,k,l,m):
    '''Function to use for finding error ranges'''
    k,x,l,m=0,0,0,0
    return k * np.exp(-(x-l)**2/m)

def err_ranges(x, func, param, sigma):
    '''The function to find error ranges for fitted data'''
   
    import itertools as iter
    
    low = func(x, *param)
    up = low
    
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        up = np.maximum(up, y)
        
    return low, up

def map_corr(df, size=6):
    '''This function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    '''

    core = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(core, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(core.columns)), core.columns, rotation=90)
    plt.yticks(range(len(core.columns)), core.columns)

    plt.colorbar()
    # no plt.show() at the end
    def n_cluster(data_frame):
        '''This function takes a data frame as an input and performs 
        the k-means clustering algorithm to determine 
        the optimal number of clusters. It returns a tuple of two lists: 
        the range of k values used and the sum of squared errors (SSE) for each value of k.'''
        k_rng = range(1,10)
        sse=[]
        for k in k_rng:
            km = KMeans(n_clusters=k)
            km.fit_predict(data_frame)
            sse.append(km.inertia_)
        return k_rng,sse


    #This code reads in data from an Excel file and  the "Year" column is converted from an object to an integer type.
    datax1 =  read_data("DATA.xls")
    warnings.filterwarnings("ignore")
    start = 1960
    end = 2015
    year = [str(i) for i in range(start, end+1)]
    Indicator = ['Aquaculture production (metric tons)','CO2 emissions from liquid fuel consumption (% of total)']
    data = stat_data(datax1,'Country Name', 'India', year , Indicator)
    Indicatorx = ['Aquaculture production (metric tons)','CO2 emissions from liquid fuel consumption (% of total)','CO2 intensity (kg per kg of oil equivalent energy use)','Capture fisheries production (metric tons)']
    datax = stat_data(datax1,'Country Name', 'India', year , Indicatorx)
    data = data.rename_axis('Year').reset_index()
    data['Year'] = data['Year'].astype('int')

    
