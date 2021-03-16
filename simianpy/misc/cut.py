import pandas as pd
import numpy as np

def cut(series, bins, label_method='centers'):
    if label_method == 'center':
        bin_centers = np.mean([bins[1:],bins[:-1]],axis=0)
        return pd.cut(series, bins=bins, labels=bin_centers)
    elif label_method == 'left':
        return pd.cut(series, bins=bins, labels=bins[:-1])
    elif label_method == 'right':
        return pd.cut(series, bins=bins, labels=bins[1:])
    else:
        raise NotImplementedError(f'unsupported label_method: {label_method}')