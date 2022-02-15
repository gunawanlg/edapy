import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from .utils import autolabel


def grid_plots(ncols, nrows, n=None, figsize=None):
    n = n or nrows * ncols
    figsize = figsize or (16, 3.5*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.reshape(-1)

    # Turn off unused axis
    delta = nrows * ncols - n
    if delta > 0:
        for ax in axes[(nrows * ncols - delta):]:
            ax.axis('off')
    return fig, axes


def plot_ecdf_numerical(series, ax, **kwargs):
    ecdf = sm.distributions.ECDF(series)
    ax.plot(ecdf.x, ecdf.y, **kwargs)
    return ax


def plot_pdf_numerical(series, **kwargs):
    return sns.histplot(series, **kwargs)


def distribution_gridplots(data, cols_num, hue=None, ncols=3, axes=None, figsize=None, type='pdf'):
    """
    Parameters
    ----------
    data : pandas.DataFrame
        dataframe without infinite values. will drop null values while plotting.
    cols_num : List[str]
        interval or ratio column in data
    hue : str, default=None
        hue in distribution plot
    ncols : int, default=3
        number of grid columns
    axes : List[matplotlib.axes.Axes]
        grid axis to iterate to
    """
    def map_plot_type(type, *args, **kwargs):
        if type == 'ecdf':
            return plot_ecdf_numerical(*args, **kwargs)
        elif type == 'pdf':
            return plot_pdf_numerical(*args, **kwargs)
        else:
            raise ValueError(
                "Unknown type of plots. Supported types are ['pdf', "
                "'ecdf']."
            )
    if axes is None:
        n = len(cols_num)
        nrows = math.ceil(n / ncols)
        fig, axes = grid_plots(nrows, ncols, n=n, figsize=figsize)
    sorted_cols_num = sorted(cols_num)  # we want it sorted for easier search

    if hue is None:
        for col, ax in zip(sorted_cols_num, axes):
            ax = map_plot_type(type, data[col], ax=ax)
            ax.set_xlabel(col)
    else:
        sorted_cols_target = sorted(data[hue].unique())
        colors = list(plt.cm.tab10(np.arange(len(sorted_cols_target))))
        for col, ax in zip(sorted_cols_num, axes):
            for t, color in zip(sorted_cols_target, colors):
                ax = map_plot_type(type, data[data[hue] == t][col].dropna(), ax=ax, color=color)
            ax.set_xlabel(col)
            ax.legend(sorted_cols_target)
    return fig, axes


def ecdf_numerical(data, cols_num, hue=None, ncols=3, axes=None, figsize=None):
    """
    Empirical Cumulative Distribution Function plot. Useful for KS-test.
    """
    return distribution_gridplots(
        data, cols_num,
        hue=hue, ncols=ncols, axes=axes, figsize=figsize,
        type='ecdf'
    )


def pdf_numerical(data, cols_num, hue=None, ncols=3, axes=None, figsize=None):
    """
    Distplot numerical column attributes in small multiple grid.
    """
    return distribution_gridplots(
        data, cols_num,
        hue=hue, ncols=ncols, axes=axes, figsize=figsize,
        type='pdf'
    )


def distplot_categorical(data, cols_cat, col_target=None, normalize=True, ncols=3,
                         sort=False, kind='bar', axes=None, figsize=None):
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
    ncols : int, default=3
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
    if axes is None:
        n = len(cols_cat)
        nrows = math.ceil(n / ncols)
        fig, axes = grid_plots(nrows, ncols, n=n, figsize=figsize)
    sorted_cols_cat = sorted(cols_cat)  # we want it sorted for easier search

    if col_target is None:
        for col, ax in zip(sorted_cols_cat, axes):
            data[col].value_counts(normalize=normalize, sort=sort).plot(ax=ax, kind=kind)
            xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in ax.get_xticklabels()]
            ax.set_xticklabels(xlabels, rotation=30, ha='right')
            ax.set_xlabel(col)
    else:
        for col, ax in zip(sorted_cols_cat, axes):
            data.groupby(col_target)[col].value_counts(normalize=normalize, sort=sort).unstack(0).plot(ax=ax, kind=kind)
            xlabels = [x.get_text()[:15]+'...' if (len(x.get_text()) > 15) else x for x in ax.get_xticklabels()]
            ax.set_xticklabels(xlabels, rotation=30, ha='right')
    plt.tight_layout()
    return fig, axes


def distplot_categorical_pretty(
    data, cols_cat, normalize=True,
    axes=None, figsize=None, ncols=5,
    sort=False, text_format="{:.2f}", alignment='right',
    color="#62AF8F", lbl_limit=12, filename='',
):
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
    ncols : int, default=3
        number of grid columns
    figsize : Tuple[int, int]
        figure size
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
    if axes is None:
        n = len(cols_cat)
        nrows = math.ceil(n / ncols)
        fig, axes = grid_plots(nrows, ncols, n=n, figsize=figsize)

    for col, ax in zip(sorted(cols_cat), axes):
        plot_data = data[col].value_counts(normalize=normalize, sort=sort, dropna=False)
        bars = ax.barh(plot_data.index, plot_data.values, align='center', height=0.8, alpha=0.7, color=color)

        max_x_value = ax.get_xlim()[1]
        distance = max_x_value * 0.01

        ticks_loc = ax.get_yticks()
        ax.set_yticks(ticks_loc)
        yticklabels = [x[:lbl_limit]+'...' if (len(x) > lbl_limit) else x for x in plot_data.index]
        ax.set_yticklabels(yticklabels)

        if alignment == 'default':
            for bar in bars:
                text = text_format.format(bar.get_width())
                text_x = bar.get_width() + distance
                text_y = bar.get_y() + bar.get_height() / 2
                ax.text(text_x, text_y, text, va='center')
        elif alignment == 'right':
            for bar in bars:
                text = text_format.format(bar.get_width())
                text_x = max_x_value
                text_y = bar.get_y() + bar.get_height() / 2
                ax.text(text_x, text_y, text, va='center')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(col)

    plt.tight_layout()
    if filename != '':
        fig.savefig(filename, dpi=100, transparent=False)
    return fig, axes


def plot_share(data, col_x, col_y, legend=None, figsize=(16, 4), stacked=True, dropna=False, color=None, reindex=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    data.groupby(col_x)[col_y].value_counts(normalize=False, dropna=dropna).unstack()\
        .reindex(reindex).plot(kind='bar', alpha=0.7, stacked=stacked, ax=axes[0], color=color)
    data.groupby(col_x)[col_y].value_counts(normalize=True, dropna=dropna).unstack()\
        .reindex(reindex).plot(kind='bar', alpha=0.7, stacked=stacked, ax=axes[1], color=color)

    for ax, norm in zip(axes, [False, True]):

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        if legend is not None:
            ax.legend(legend, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                      fancybox=False, shadow=False, ncol=3)
        else:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                      fancybox=False, shadow=False, ncol=3)
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        autolabel(ax, normalized=norm)
    plt.show()


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

    plt.pcolormesh(Week, Weekday, df_pivot.values, cmap="Blues", edgecolor="w")
    plt.xlim(0, df_pivot.shape[1])
    plt.suptitle(suptitle, fontsize=20, ha='left', x=0.125)
    plt.title(title, fontsize=14, loc='left')
    plt.colorbar()
    plt.show()
