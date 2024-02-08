from itertools import combinations
import numpy as np
import pandas as pd

def pairwise_difference(data):
    # data.columns
    diff = np.expand_dims(data.values, 2) \
        - np.expand_dims(data.values, 1)
    x, y, _ = diff.shape
    a, b = np.triu_indices(y, 1)
    diff = diff[
        np.repeat(np.arange(x), a.size), 
        np.tile(a, x), 
        np.tile(b, x)
    ]
    labels = list(combinations(data.columns.values, 2)) * x
    diffdata = [(a,b,c) for (a,b), c in zip(labels, diff)]

    return pd.DataFrame(
        diffdata, 
        columns=['A', 'B', 'diff'], 
        index=np.repeat(data.index, a.size)
    )


