from typing import Literal

import numpy as np
import pandas as pd


def cut(
    x, bins, label_method: Literal["center", "left", "right", "range"] = "center"
):
    """Pandas cut function with custom labels.

    Parameters
    ----------
    x : array-like
        values to be binned
    bins : array-like
        Criteria to bin data.
    label_method : Literal["center", "left", "right", "range"], optional
        Methods to determine custom label, by default "center"
        "left" : label is the left edge of the bin
        "right" : label is the right edge of the bin
        "range" : label is the range of the bin as a string
        "center" : label is the mean of the bin edges

    Returns
    -------
    out: Categorical
        Categorical for `label_method` == "center", Series otherwise

    Raises
    ------
    NotImplementedError
        If label_method is not implemented.
    """
    if label_method is None:
        return pd.cut(x, bins=bins)
    elif label_method == "center":
        bin_centers = np.mean([bins[1:], bins[:-1]], axis=0)
        return pd.cut(x, bins=bins, labels=bin_centers)
    elif label_method == "left":
        return pd.cut(x, bins=bins, labels=bins[:-1])
    elif label_method == "right":
        return pd.cut(x, bins=bins, labels=bins[1:])
    elif label_method == "range":
        labels = [f"{left}-{right}" for left, right in zip(bins[:-1], bins[1:])]
        return pd.cut(x, bins=bins, labels=labels)
    else:
        raise NotImplementedError(f"unsupported label_method: {label_method}")
