import numpy as np
import pandas as pd


def cut(series, bins, label_method="center"):
    if label_method is None:
        return pd.cut(series, bins=bins)
    elif label_method == "center":
        bin_centers = np.mean([bins[1:], bins[:-1]], axis=0)
        return pd.cut(series, bins=bins, labels=bin_centers)
    elif label_method == "left":
        return pd.cut(series, bins=bins, labels=bins[:-1])
    elif label_method == "right":
        return pd.cut(series, bins=bins, labels=bins[1:])
    elif label_method == "range":
        bins = [f"{left}-{right}" for left, right in zip(bins[:-1], bins[1:])]
        return pd.cut(series, bins=bins, labels=bins)
    else:
        raise NotImplementedError(f"unsupported label_method: {label_method}")
