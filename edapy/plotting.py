import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

def distplot_numerical(data, cols_num, col_target=None):
    if col_target is None:
        n = math.ceil(len(cols_num) / 3)
        fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
        for col, a in zip(cols_num, ax.reshape(-1)):
            sns.distplot(data[col], ax=a)
            a.set_xlabel(col)
    else:
        if (data[col_target].nunique() == 2): # binary target
            target_mask = data[col_target] == data[col_target].unique()[0]
            n = math.ceil(len(cols_num) / 3)
            fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
            for col, a in zip(cols_num, ax.reshape(-1)):
                sns.distplot(data[target_mask][col].dropna(), ax=a)                                    
                sns.distplot(data[~target_mask][col].dropna(), ax=a)
                a.legend(data[col_target].unique())
                a.set_xlabel(col)
        else: # regression target, plot only distribution of attributes
            n = math.ceil(len(cols_num) / 3)
            fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
            for col, a in zip(cols_num, ax.reshape(-1)):
                sns.distplot(data[col], ax=a)
                a.set_xlabel(col)

def distplot_categorical(data, cols_cat, col_target=None, normalize=True):
    if col_target is None:
        n = math.ceil(len(cols_cat) / 3)
        fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
        for col, a in zip(cols_cat, ax.reshape(-1)):
            data[col].value_counts(normalize=normalize).plot.bar(ax=a)
            a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right')
            a.set_xlabel(col)
        plt.tight_layout()
    else:        
        if (data[col_target].nunique() == 2): # binary target
            n = math.ceil(len(cols_cat) / 3)
            fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
            for col, a in zip(cols_cat, ax.reshape(-1)):
                data.groupby(col_target)[col].value_counts(normalize=normalize).unstack(0).plot.bar(ax=a)
                a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right')
            plt.tight_layout()
        else: # regression target, distribution of target by each attributes                
            n = math.ceil(len(cols_cat) / 3)
            fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
            for col, a in zip(cols_cat, ax.reshape(-1)):
                for val in data[col].unique():
                    sns.distplot(data[data[col] == val][col_target], ax=a)
                a.legend(data[col].unique())
                a.set_xlabel(col)
            plt.tight_layout()

def diff_distplot_numerical(data, cols_num, col_target, filt_idx):
    if (data[col_target].nunique() == 2): # binary target
        target_mask = data[col_target] == data[col_target].unique()[0]
        n = math.ceil(len(cols_num) / 3)
        fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
        for col, a in zip(cols_num, ax.reshape(-1)):
            sns.distplot(data[target_mask][col].dropna(), ax=a)
            sns.distplot(data[~target_mask][col].dropna(), ax=a)
            sns.distplot(data[filt_idx & target_mask][col].dropna(), ax=a)
            sns.distplot(data[filt_idx & ~target_mask][col].dropna(), ax=a)
            a.legend(list(data[col_target].unique()) + ['filt_no', 'filt_yes'])
            a.set_xlabel(col)

def diff_distplot_categorical(data, cols_cat, col_target, filt_idx):
    if (data[col_target].nunique() == 2): # binary target
        n = math.ceil(len(cols_cat) / 3)
        fig, ax = plt.subplots(n, 3, figsize=(15, 3.5*n))
        for col, a in zip(cols_cat, ax.reshape(-1)):
            data1 = data.groupby(col_target)[col].value_counts(normalize=True).unstack(0)
            data2 = data[filt_idx].groupby(col_target)[col].value_counts(normalize=True).unstack(0)
            (data2 - data1).plot.bar(ax=a)
            a.set_ylim([-1.0, 1.0])
            a.axhline(color='gray')
            a.set_xticklabels(a.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()