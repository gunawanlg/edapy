import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def ecdf_numerical(data, cols_num, col_target=None, grid_c=3, w=15, h_factor=3.5):
    """
    Empirical Cumulative Distribution Function plot. Useful for KS-test.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_num : list of str
        interval or ratio column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    """
    n = math.ceil(len(cols_num) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    sorted_cols_num = sorted(cols_num)  # we wnat it sorted for easier search
    
    if col_target is None:        
        for col, a in zip(sorted_cols_num, ax.reshape(-1)):
            ecdf = sm.distributions.ECDF(data[col])
            ax.plot(ecdf.x, ecdf.y)
            a.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[col_target].unique())
        if len(sorted_cols_target) > 1 and len(sorted_cols_target) <= 5:  # > 5 will be too crowded
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                for t in sorted_cols_target:
                    ecdf = sm.distributions.ECDF(data[data[col_target] == t][col].dropna())
                    a.plot(ecdf.x, ecdf.y)
                a.legend(sorted_cols_target)
                a.set_xlabel(col)
        else:  # most probably regression analysis
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                ecdf = sm.distributions.ECDF(data[col].dropna())
                a.plot(ecdf.x, ecdf.y)
                a.set_xlabel(col)


def distplot_numerical(data, cols_num, col_target=None, grid_c=3, w=15, h_factor=3.5):
    """
    Distplot numerical column attributes in small multiple grid.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_num : list of str
        interval or ratio column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    """
    n = math.ceil(len(cols_num) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    sorted_cols_num = sorted(cols_num)  # we wnat it sorted for easier search
    
    if col_target is None:        
        for col, a in zip(sorted_cols_num, ax.reshape(-1)):
            sns.distplot(data[col], ax=a)
            a.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[col_target].unique())
        if len(sorted_cols_target) > 1 and len(sorted_cols_target) <= 5:  # > 5 will be too crowded
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                for t in sorted_cols_target:
                    sns.distplot(data[data[col_target] == t][col].dropna(), ax=a)
                a.legend(sorted_cols_target)
                a.set_xlabel(col)
        else:  # most probably regression analysis
            for col, a in zip(sorted_cols_num, ax.reshape(-1)):
                sns.distplot(data[col], ax=a)
                a.set_xlabel(col)


def distplot_categorical(data, cols_cat, col_target=None, normalize=True, grid_c=3, w=15, 
                         h_factor=3.5, sort=False, kind='bar'):
    """
    Distplot categorical column attributes in small multiple grid.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_cat : list of str
        categorical column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    normalize : bool, default=True
        wether to normalize the count or not
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    sort : bool, default=False
        prevent sorting based on counts, will fallback to .cat.categories if the series is having
        category dtype
    kind : str, default='bar'
        matplotlib plot kind, really recommend to do bar plot, alternative would be 'barh'
    """
    n = math.ceil(len(cols_cat) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    sorted_cols_cat = sorted(cols_cat)  # we want it sorted for easier search

    if col_target is None:        
        for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
            data[col].value_counts(normalize=normalize, sort=sort).plot(ax=a, kind=kind)
            xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
            a.set_xticklabels(xlabels, rotation=30, ha='right')
            a.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[col_target].unique())
        if len(sorted_cols_target) > 1 and len(sorted_cols_target) <= 5:  # > 5 will be too crowded
            for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
                data.groupby(col_target)[col].value_counts(normalize=normalize, sort=sort).unstack(0).plot(ax=a, kind=kind)
                xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
                a.set_xticklabels(xlabels, rotation=30, ha='right')
        else:  # most probably regression analysis
            for col, a in zip(sorted_cols_cat, ax.reshape(-1)):
                data[col].value_counts(normalize=normalize, sort=sort).plot(ax=a, kind=kind)
                xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in a.get_xticklabels()]
                a.set_xticklabels(xlabels, rotation=30, ha='right')
                a.set_xlabel(col)
    plt.tight_layout()


def diff_distplot_numerical(data, cols_num, col_target, filt_idx, grid_c=3, w=15, h_factor=3.5):
    """
    Difference distplot numerical column attributes in small multiple grid.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_num : list of str
        interval or ratio column in data
    col_target : str
        the target variable we want to distingusih the cols_num distribution
    filt_idx : list of bool
        boolean indexing filter
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
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

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_cat : list of str
        categorical column in data
    col_target : str, optional
        the target variable we want to distingusih the cols_num distributino
    filt_idx : list of bool
        boolean indexing filter    
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
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


def waffle_chart(df_pivot, suptitle='', title='', figsize=(14, 2.8)): 
    """
    Create waffle chart like the one in github contribution.

    Parameters
    ----------
    df_pivot : pandas.DataFrame
        pivot dataframe with 3 columns (x, y, values)
    suptitle : str, default=''
        title string in the plot
    title : str, default = ''
        subtitle string in the plot
    figsize : (float, float), default=(14, 2.8)
        figisze arguments of plt.subplots()
    """
    Weekday, Week = np.mgrid[:df_pivot.shape[0]+1, :df_pivot.shape[1]+1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_yticklabels('')
    ax.set_yticks(np.array(range(len(df_pivot.index))) + 0.5, minor=True)
    ax.set_yticklabels(list(df_pivot.index), minor=True)
    ax.set_xticks(list(range(df_pivot.shape[1])))
    ax.set_xticklabels(list(df_pivot.columns))
    ax.set_xlabel(df_pivot.columns.name)
    
    plt.pcolormesh(Week, Weekday, df_pivot.values, cmap="Blues", edgecolor="w", vmin=-10, vmax=100)
    plt.xlim(0, df_pivot.shape[1])
    plt.suptitle(suptitle,fontsize=20, ha='left', x=0.125)
    plt.title(title,fontsize=14, loc='left')
    plt.show()


def distplot_categorical_pretty(data, cols_cat, normalize=True, grid_c=5, w=15, h_factor=3.5,
                                sort=False, text_format="{:.2f}", alignment='right', filename='',
                                color="#62AF8F", lbl_limit=12):
    """
    Plot binned numerical or categorical column vertically with text percentage shown.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values, will drop null values while plotting
    cols_cat : list of str
        categorical column in data
    normalize : bool, default=True
        wether to normalize the counts or not, ideally is True
    grid_c : int, default=3
        number of grid columns
    w : int, default=15
        figsize witdh arguments
    h_factor : float, default=3.5
        height of small plot
    sort : bool, default=False
        prevent plot sorting from smallest count, wii fallback to .cat.categories if column
        dtype is categorical
    text_format : str, default={:.2f}
        python string formatting
    alignment : str, default='right'
        if 'default', the text percentage will be near the bar plotted
    filename : str, default=''
        if '', will not save the plot
    color : str, default="#62AF8F"
        hex format string of bar color
    lbl_limit : int, default=12
        label limit, will truncate longer label
    """
    n = math.ceil(len(cols_cat) / grid_c)
    fig, ax = plt.subplots(n, grid_c, figsize=(w, h_factor*n))
    plt.draw()
    for col, a in zip(sorted(cols_cat), ax.reshape(-1)):
        plot_data = data[col].value_counts(normalize=normalize, sort=sort)
        bars = a.barh(plot_data.index, plot_data.values, align='center', height=0.8, alpha=0.7, color=color)
        
        max_x_value = a.get_xlim()[1]
        distance = max_x_value * 0.01
        
        yticklabels = [x[:lbl_limit]+'...' if (len(x) > lbl_limit) else x for x in plot_data.index]
        a.set_yticklabels(yticklabels)

        if alignment == 'default':
            for bar in bars:
                text = text_format.format(bar.get_width())
                text_x = bar.get_width() + distance
                text_y = bar.get_y() + bar.get_height() / 2
                a.text(text_x, text_y, text, va='center')
        elif alignment == 'right':
            for bar in bars:
                text = text_format.format(bar.get_width())
                text_x = max_x_value
                text_y = bar.get_y() + bar.get_height() / 2
                a.text(text_x, text_y, text, va='center')
        
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_title(col)
        
    plt.tight_layout()
    if filename is not '':
        plt.savefig(filename, dpi=100, transparent=False)