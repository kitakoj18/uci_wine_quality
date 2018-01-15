'''
Red wine analysis from UCI dataset:
https://archive.ics.uci.edu/ml/datasets/wine+quality
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_outlier_ind(values, upper_outliers = True):
    '''
    values(array): array of feature values to use to determine quartile ranges and outliers
    upper_outliers(bool): True to find only outliers above distribution, False to find below distribution

    Returns array of indices where outliers are found for particular feature
    '''

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    if upper_outliers:
        outlier_thres = q3 + (1.5 * iqr)
        return np.argwhere(values >= outlier_thres)

    else:
        outlier_thres = q1 - (1.5 * iqr)
        return np.argwhere(values <= outlier_thres)

def get_outlier_resp_values(resp_values, outlier_ind):
    '''
    resp_values(array): array of response variable values
    outlier_ind(array): array of outlier indices

    Returns response variable values of outliers
    '''

    vals = resp_values[outlier_ind]
    return np.reshape(vals, len(vals))

if __name__ == '__main__':

    #data separated with ;
    winedf = pd.read_csv('../data/winequality-red.csv', sep = ';')

    #columns
    wine_cols = winedf.columns

    '''
    EDA
    '''

    #make sure data imported correctly
    print(winedf.head())


    #check any null values
    print(winedf.info())
    #confirmed - clean dataset, no null values for any variables


    #correlation between variables
    fig1 = plt.figure(figsize = (15, 15))
    sns.heatmap(winedf.corr(), annot = True, linewidths = 0.5, cmap = 'Blues')
    #plt.show()
    plt.savefig('../images/correlations.png')


    #boxplots of each variable
    fig2 = plt.figure(figsize = (20, 10))
    fig2.suptitle('Distributions of Wine Features', fontsize = 34, fontweight = 'bold')
    #create a boxplot for each feature in same figure
    for i in range(len(wine_cols)):
        plt.subplot(3, 4, i+1)
        sns.boxplot(x = winedf[wine_cols[i]])
    #plt.show()
    plt.savefig('../images/feature_boxplots.png')


    #look to see if outliers for each feature have similar response values
    #use outlier functions to get indices of outliers and their respective response variable values
    feat_cols = winedf.columns[:11]
    for col in feat_cols:
        print(col)
        feat_vals = winedf[col].values
        out_ind = get_outlier_ind(feat_vals)
        out_res = get_outlier_resp_values(winedf['quality'].values, out_ind)

        #quality values range from 0-8 so match with their respective counts as determined by functions
        #bincount creates count for each value
        for val, val_count in zip(np.arange(9), np.bincount(out_res)):
            print('Quality Value: {} || Count: {} /n'.format(val, val_count))

        print('/n/n')

    #look at relationship between features and quality ratings
    fig3 = plt.figure(figsize= (25, 15))
    fig3.suptitle('Relationship Between Features and Quality Ratings', fontsize = 28, fontweight = 'bold')
    #create a scatterplot for each feature vs. quality rating
    for i in range(len(feat_cols)):
        ax = fig3.add_subplot(3, 4, i+1)
        ax.set_title('{} vs. {}'.format(feat_cols[i], 'quality'), fontsize = 10, fontweight = 'bold')
        ax.set_xlabel(feat_cols[i], fontsize = 8, fontweight = 'bold')
        ax.set_ylabel('quality', fontsize = 8, fontweight = 'bold')
        plt.scatter(x = winedf[feat_cols[i]], y = winedf['quality'])
    #plt.show()
    plt.savefig('../images/features_vs_quality.png')

    #see distribution of quality ratings
    fig4 = plt.figure()
    fig4.suptitle('Quality Rating Counts', fontweight = 'bold')
    ax = fig4.add_subplot(1, 1, 1)
    ax.set_xlabel('Quality Rating', fontweight = 'bold')
    ax.set_ylabel('Quality Count', fontweight = 'bold')
    winedf['quality'].value_counts().sort_index().plot(kind = 'bar')
    #plt.show()
    plt.savefig('../images/quality_rating_counts.png')


    '''
    Modeling
    '''
