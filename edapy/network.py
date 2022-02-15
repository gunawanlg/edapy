from collections import Counter
from .utils import get_unique_tuple
from tqdm.auto import tqdm


def get_complete_edges(df, cols):
    """
    Arguments
    ---------
        df {pd.DataFrame} -- dataframe of contract
        cols {list} -- list of attributes to be checked to
    Return
    ------
        d =
        {
            'key': [[], [], [], [], ...], # len = number of total data
            'key1': [[], [], [], [], ...],
            ...
        }
        d[key][0] represent vertices connected to data at index 0 with `CONNECTION_TYPE` of `key`.

    """
    indexes = range(df.shape[0])
    d = {}
    for col in cols:
        targets = []
        for n in indexes:
            source = n
            same_val = (df[col] == df.loc[source, col]).values
            target = [i for i, x in enumerate(same_val) if x]
            try:
                target.remove(source)
            except Exception as e:
                print(e)
                pass
            targets.append(target)
        d[col] = targets
    return d

# NEED OPTIMIZATION O(n^2)


def get_tuple_edges(df, cols, directed=False, weight=False):
    """
    Arguments
    ---------
        df {pd.DataFrame} -- dataframe of ews dataset
        cols {list} -- column names of attributes to be checked
    Keyword Arguments
    -----------------
        directed {bool} -- True to create directed graph, False to create undirected graph
        weight {bool} -- True to consider all attributes as same, combining it as weight of edges
    Return
    ------
        d {dict} -- dictionary with key of cols, and value of list of tuples of edges (e.g [(1, 2), (2, 3), ...])
    """
    d = {}
    weights = []  # calculates weight by counting number of connections
    for col in tqdm(cols):
        targets = []

        # Null value data
        XNA_mask = df[col].isnull()
        if (df[col].dtype in ['O', 'str']):
            XNA_mask = XNA_mask | (df[col] == 'XNA')

        # Mark all duplicates as True, keep=False
        indexes = list(df[~XNA_mask & df[col].duplicated(keep=False)].index)

        for n in tqdm(indexes):
            source = n
            same_val = (df[col] == df.loc[source, col])
            target = [(source, i) for i, x in same_val.items() if x]
            try:
                target.remove((source, source))
            except Exception as e:
                print(e)
                pass
            targets.extend(target)

        if directed is False:  # Create undirected graph: (1, 2) is the same as (2, 1)
            targets = get_unique_tuple(targets)
        if weight is True:
            weights.extend(targets)
        else:
            d[col] = targets

    if weight is True:
        c = Counter(x for x in weights)
        d['connection'] = [(x, y, val) for (x, y), val in c.items()]
    return d
