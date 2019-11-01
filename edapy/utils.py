import numpy as np
import pandas as pd

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_unique_tuple(a):
    return [tuple(x) for x in set(map(frozenset, a))]

def lookup_date(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    Example: df['date'] = lookup_date(df['date'])
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)    