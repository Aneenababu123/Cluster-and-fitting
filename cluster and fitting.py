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
# %matplotlib inline


def read_data(filename):
    '''
    This is a function called read_data that takes a filename as an input
    '''
    df = pd.read_excel(filename, skiprows=3)
    return df


def stat_data(df, col, value, yr, a):
    """This function takes a pandas DataFrame df, a column name col, a value to filter on value
    """
    df3 = df.groupby(col, group_keys=True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    df3 = df3.transpose()
    df3 = df3.loc[:, a]
    df3 = df3.dropna(axis=1)
    return df3


def Expo(t, scale, growth):
    ''' The function calculates the value of an exponential function at a given time point, using the specified scaling and growth factors'''
    f = (scale * np.exp(growth * (t-1960)))
    return f


def func(x, k, l, m):
    '''Function to use for finding error ranges'''
    k, x, l, m = 0, 0, 0, 0
    return k * np.exp(-(x-l)**2/m)


def err_ranges(x, func, param, sigma):
    '''The function to find error ranges for fitted data'''

    import itertools as iter

    low = func(x, *param)
    up = low

    uplow = []
    for p, s in zip(param, sigma):
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
        k_rng = range(1, 10)
        sse = []
        for k in k_rng:
            km = KMeans(n_clusters=k)
            km.fit_predict(data_frame)
            sse.append(km.inertia_)
        return k_rng, sse

    # This code reads in data from an Excel file and  the "Year" column is converted from an object to an integer type.
    datax1 = read_data("DATA.xls")
    warnings.filterwarnings("ignore")
    start = 1960
    end = 2015
    year = [str(i) for i in range(start, end+1)]
    Indicator = ['Aquaculture production (metric tons)',
                 'CO2 emissions from liquid fuel consumption (% of total)']
    data = stat_data(datax1, 'Country Name', 'India', year, Indicator)
    Indicatorx = ['Aquaculture production (metric tons)', 'CO2 emissions from liquid fuel consumption (% of total)',
                  'CO2 intensity (kg per kg of oil equivalent energy use)', 'Capture fisheries production (metric tons)']
    datax = stat_data(datax1, 'Country Name', 'India', year, Indicatorx)
    data = data.rename_axis('Year').reset_index()
    data['Year'] = data['Year'].astype('int')
    # This code block is performing feature scaling on two columns of the 'data' dataframe using MinMaxScaler.
    scaler = MinMaxScaler()
    scaler.fit(data[['Aquaculture production (metric tons)']])
    data['Scaler_A'] = scaler.transform(
        data['Aquaculture production (metric tons)'].values.reshape(-1, 1))

    scaler.fit(
        data[['CO2 emissions from liquid fuel consumption (% of total)']])
    data['Scaler_C'] = scaler.transform(
        data['CO2 emissions from liquid fuel consumption (% of total)'].values.reshape(-1, 1))
    data_c = data.loc[:, ['Scaler_A', 'Scaler_C']]

    # This code defines a curve fitting process using the curve_fit() and the fitted curve is then calculated using the Expo() function with the fitted parameters and the Year column from the data.
    popt, pcov = opt.curve_fit(
        Expo, data['Year'], data['Aquaculture production (metric tons)'], p0=[1000, 0.02])
    data["Pop"] = Expo(data['Year'], *popt)
    sigma = np.sqrt(np.diag(pcov))
    low, up = err_ranges(data["Year"], Expo, popt, sigma)
    data2 = data
    # This code is creating a scatter plot
    plt.figure()
    sns.scatterplot(data=data2, x="Year",
                    y="Aquaculture production (metric tons)", cmap="Accent")
    plt.title('Scatter Plot between 1990-2015 before fitting')
    plt.ylabel('Trade (% of GDP)')
    # plt.xlabel('Year')
    plt.xlim(1990, 2015)
    plt.savefig("Scatter_fit.png")

    # The code you provided is plotting a graph to visualize the data and the fitted exponential function
    plt.figure()
    plt.title("Plot After Fitting")
    plt.plot(data["Year"],
             data['Aquaculture production (metric tons)'], label="data")
    plt.plot(data["Year"], data["Pop"], label="fit")
    plt.fill_between(data["Year"], low, up, alpha=0.7)
    plt.legend()
    plt.xlabel("Year")

    # Corelation plot
    plt.figure()
    corr = datax.corr()
    map_corr(datax)
    plt.title("corelation")
    plt.savefig("corelation plot.png")

    # Scatter matrix plot
    plt.figure()
    pd.plotting.scatter_matrix(datax, figsize=(9, 9))
    plt.tight_layout()
    plt.title('Scatter plot')
    plt.savefig("Scatter plot.png")

    # Scatter plot
    plt.figure()
    plt.scatter(data['Aquaculture production (metric tons)'],
                data['CO2 emissions from liquid fuel consumption (% of total)'])
    plt.title("Scatter plot")
    plt.savefig('Plot.png')

    # Plotting the sum of squared error
    plt.figure()
    a, b = n_cluster(data_c)
    # plt.xlabel=('K')
    plt.ylabel('sum of squared error')
    plt.plot(a, b)
    plt.title('sum of squared error')
    plt.savefig('Squared error.png')

    # performs K-means clustering on the data frame and generate cluster predictions and resulting clusters are then plotted in a scatter plot
    km = KMeans(n_clusters=4)
    pred = km.fit_predict(data_c[['Scaler_A', 'Scaler_C']])
    data_c['cludter'] = pred
    # data_c.head()
    centers = km.cluster_centers_
    dc1 = data_c[data_c.cludter == 0]
    dc2 = data_c[data_c.cludter == 1]
    dc3 = data_c[data_c.cludter == 2]
    dc4 = data_c[data_c.cludter == 3]
    plt.figure()
    plt.scatter(dc1['Scaler_A'], dc1['Scaler_C'], color='green')
    plt.scatter(dc2['Scaler_A'], dc2['Scaler_C'], color='red')
    plt.scatter(dc3['Scaler_A'], dc3['Scaler_C'], color='blue')
    plt.scatter(dc4['Scaler_A'], dc4['Scaler_C'], color='yellow')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', color='black')
    plt.title("Cluster Plot")
    plt.savefig('Cluster plot.png')
    plt.show()
