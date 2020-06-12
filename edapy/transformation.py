import numpy as np
import pandas as pd

def convert_to_categorical(df, cat_limit=20):
    for col in df.columns:
        if (df[col].nunique() <= cat_limit):
            df[col] = df[col].astype('category')
            print("Column {} casted to categorical".format(col))


def reduce_ordinal_category(series, bins, values):
    temp = pd.cut(series, bins=bins).astype('str')
    keys = temp.unique()
    mapper = dict(zip(keys, values))
    return temp.map(mapper)


def outlier_removal(X, method='Tukey', k=3):
    Q3 = X.quantile(0.75)
    Q1 = X.quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 3*IQR
    lower = Q1 - 3*IQR
    res = (X < lower) | (X > upper)
    return res


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def create_pivot(data, x, y):
    g = data.groupby([y, x], as_index=False).size().reset_index(name='count')
    df = g.pivot(columns=x, index=y, values="count")
    df.fillna(0, inplace=True)
    df = df.reindex(delays_mean.index)[:5]
    df = df[::-1]
    return df