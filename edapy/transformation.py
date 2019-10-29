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