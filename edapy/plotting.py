import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

def distplot_numerical(data, cols_num, col_target=None, grid_c=3, w=15, h_factor=3.5):
    """
    Distplot numerical column attributes in small multiple grid.

    Arguments:
        data {pd.DataFrame} -- dataframe without infinite values
        cols_num [str] -- numerical column in dataframe

    Keyword Arguments:
        col_target {str} -- if None, plot distribution without col_target grouping
                             else plot cols_num attribute depending on type of col_target unique values,
                             same as None if col_target unique values is not equal to 2
        grid_c {int} -- default 3, number of column grid
        w {int} -- default 15, figsize width arguments
        h_factor {float} -- default 3.5, height of small plots
    """
    n = math.ceil(len(cols_num) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if col_target is None:        
        for col, a in zip(sorted(cols_num), ax.reshape(-1)):
            sns.distplot(data[col], ax=a)
            a.set_xlabel(col)
    else:
        if (data[col_target].nunique() == 2): # binary target
            target_mask = data[col_target] == sorted(data[col_target].unique())[0]
            for col, a in zip(sorted(cols_num), ax.reshape(-1)):
                sns.distplot(data[target_mask][col].dropna(), ax=a)                                    
                sns.distplot(data[~target_mask][col].dropna(), ax=a)
                a.legend(sorted(data[col_target].unique()))
                a.set_xlabel(col)
        else: # regression target, plot only distribution of attributes
            for col, a in zip(sorted(cols_num), ax.reshape(-1)):
                sns.distplot(data[col], ax=a)
                a.set_xlabel(col)

def distplot_categorical(data, cols_cat, col_target=None, normalize=True, grid_c=3, w=15, h_factor=3.5):
    """
    Distplot categorical column attributes in small multiple grid.

    Arguments:
        data {pd.DataFrame} -- dataframe without infinite values
        cols_cat [str] -- categorical column in dataframe

    Keyword Arguments:
        col_target {str} -- if None, plot barchart using df.plot.bar without col_target grouping
                             else plot cols_num attribute depending on type of col_target unique values,
                             if binary col_target, plot barchart using df.plot.bar grouped by col_target,
                             if numerical col_target, plot distplot using sns.displot grouped by col_target
        normalize {Boolean} -- True to plot normalized values
        grid_c {int} -- default 3, number of column grid
        w {int} -- default 15, figsize width arguments
        h_factor {float} -- default 3.5, height of small plots
    """
    n = math.ceil(len(cols_cat) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if col_target is None:
        for col, a in zip(sorted(cols_cat), ax.reshape(-1)):
            data[col].value_counts(normalize=normalize).plot.bar(ax=a)
            a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right')
            a.set_xlabel(col)
        plt.tight_layout()
    else:        
        if (data[col_target].nunique() == 2): # binary target
            for col, a in zip(sorted(cols_cat), ax.reshape(-1)):
                data.groupby(col_target)[col].value_counts(normalize=normalize).unstack(0).plot.bar(ax=a)
                xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
                a.set_xticklabels(xlabels, rotation=30, ha='right')
            plt.tight_layout()
        else: # regression target, distribution of target by each attributes                
            for col, a in zip(sorted(cols_cat), ax.reshape(-1)):
                for val in data[col].unique():
                    sns.distplot(data[data[col] == val][col_target], ax=a)
                a.legend(data[col].unique())
                a.set_xlabel(col)

def diff_distplot_numerical(data, cols_num, col_target, filt_idx, grid_c=3, w=15, h_factor=3.5):
    """
    Difference distplot numerical column attributes in small multiple grid.

    Arguments:
        data {pd.DataFrame} -- dataframe without infinite values
        cols_num [str] -- numerical column in dataframe

    Keyword Arguments:
        col_target {str} -- if binary col_target, make 4 distplot combination of col_target and filt_idx unique values
        filt_idx [Boolean] -- boolean list, used to filter data by using boolean indexing
        grid_c {int} -- default 3, number of column grid
        w {int} -- default 15, figsize width arguments
        h_factor {float} -- default 3.5, height of small plots
    """
    n = math.ceil(len(cols_num) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if (data[col_target].nunique() == 2): # binary target
        sorted_keys = sorted(list(data[col_target].unique()))
        target_mask = data[col_target] == sorted_keys[0]
        for col, a in zip(sorted(cols_num), ax.reshape(-1)):
            sns.distplot(data[target_mask][col].dropna(), ax=a)
            sns.distplot(data[~target_mask][col].dropna(), ax=a)
            sns.distplot(data[filt_idx & target_mask][col].dropna(), ax=a)
            sns.distplot(data[filt_idx & ~target_mask][col].dropna(), ax=a)
            a.legend(sorted_keys + ['filt_'+sorted_keys[0], 'filt_'+sorted_keys[1]])
            a.set_xlabel(col)

def diff_distplot_categorical(data, cols_cat, col_target, filt_idx, grid_c=3, w=15, h_factor=3.5):
    """
    Difference distplot categorical column attributes in small multiple grid.

    Arguments:
        data {pd.DataFrame} -- dataframe without infinite values
        cols_cat [str] -- numerical column in dataframe

    Keyword Arguments:
        col_target {str} -- if binary col_target, make diff barplot with negative and positive normalized value_counts
        filt_idx [Boolean] -- boolean list, used to filter data by using boolean indexing
        grid_c {int} -- default 3, number of column grid
        w {int} -- default 15, figsize width arguments
        h_factor {float} -- default 3.5, height of small plots
    """
    n = math.ceil(len(cols_cat) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    if (data[col_target].nunique() == 2): # binary target
        for col, a in zip(sorted(cols_cat), ax.reshape(-1)):
            data1 = data.groupby(col_target)[col].value_counts(normalize=True).unstack(0)
            data2 = data[filt_idx].groupby(col_target)[col].value_counts(normalize=True).unstack(0)
            (data2 - data1).plot.bar(ax=a)
            a.set_ylim([-1.0, 1.0])
            a.axhline(color='gray')
            xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
            a.set_xticklabels(xlabels, rotation=30, ha='right')
        plt.tight_layout()