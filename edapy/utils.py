import numpy as np
import pandas as pd


def batch(l, b=1, n=None):
    """
    Create batch from iterable.

    Parameters
    ----------
    l : list
        list to create batches from
    b : int, optional, [default=1]
        batch size
    n : int, optional, [default=None]
        if None: len(batches[-1]) < b if len(iterable) % b != 0
        else: len(batches[-1]) == b if len(iterable) % b == 0
        this will override b param

    Returns
    -------
    batches : iterable
        generator of batch

    Example
    -------
    If n is None, or not inputted
    >>> l=list(range(10))
    >>> batches=batch(l, 3)
    >>> batches
    <generator object batch at 0x005A0370>
    >>> for b in batches:
    ...    print(b)
    ...
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]

    if n is not None:
    >>> l=list(range(10))
    >>> batches=batch(l, n=3)
    >>> batches
    <generator object batch at 0x006C0F30>
    >>> for b in batches:
    ...    print(b)
    ...
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8, 9]
    """
    if n is not None:
        assert n > 0
        b = int(len(l) / n)
        for ndx in range(0, n - 1):
            yield l[ndx * b:ndx * b + b]
        yield l[n * b - b:]
    else:
        assert b > 0
        m = len(l)
        for ndx in range(0, m, b):
            yield l[ndx:min(ndx + b, m)]


def get_unique_tuple(a):
    return [tuple(x) for x in set(map(frozenset, a))]


def lookup_date(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    Example: df['date']=lookup_date(df['date'])
    """
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


def sample_numpy(A, n, replace=False):
    """
    Sample numpy array.

    Parameters
    ----------
    A : numpy.ndarray(shape=[m, *dims])
        numpy array to be sampled from, with number of examples in first
        dimension
    n : int
        number of samples
    replace : bool
        False to perform sampling without replacement

    Returns
    -------
    sample : numpy.ndarray(shape=[n, *dims])
        sampled numpy array from A
    """
    return A[np.random.choice(A.shape[0], n, replace=replace), :]


def trunc_string(s, limit=100):
    """Aku adalah dia --> Aku ada..."""
    return s[:limit]+'...' if len(s) > limit else s


def autolabel(ax, normalized=True, barh=False, **kwargs):
    # Attach some text labels.
    for rect in ax.patches:
        if barh:
            if normalized:
                ax.text(rect.get_x() + rect.get_width() / 2.,
                        rect.get_y() + rect.get_height() / 2.,
                        '{}'.format(str(round(rect.get_width() * 100, 2)) + '%'),
                        ha='center',
                        va='center',
                        **kwargs)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2.,
                        rect.get_y() + rect.get_height() / 2.,
                        '{:.0f}'.format(rect.get_width()),
                        ha='center',
                        va='center',
                        **kwargs)
        else:
            if normalized:
                ax.text(rect.get_x() + rect.get_width() / 2.,
                        rect.get_y() + rect.get_height() / 2.,
                        '{}'.format(str(round(rect.get_height() * 100, 2)) + '%'),
                        ha='center',
                        va='center',
                        **kwargs)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2.,
                        rect.get_y() + rect.get_height() / 2.,
                        '{:.0f}'.format(rect.get_height()),
                        ha='center',
                        va='center',
                        **kwargs)
