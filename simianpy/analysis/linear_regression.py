import scipy.stats
import numpy as np
import pandas as pd 

class LinearRegression:
    """ Compute the linear regression of two variables

    Parameters
    ----------
    x, y: array-like or str
        these must be array-like of numeric with equal length
        or refer to keys for numeric variables in a dataframe-like object
    data: dataframe-like or None; default=None
        If provided, 'x' and 'y' are used as column indexes in data
    drop_na: bool; default=True
        If true, cases with nan values are excluded

    Attributes
    ----------
    data: pd.DataFrame
        data used for computing linear regression 
    m, b, r, p, e: numeric
        outputs of scipy.stats.linregress
    x_pred, y_pred: np.ndarray
        coordinates for line of best fit
    x_label, y_label: str
        labels for x and y variables
    """
    def __init__(self, x, y, data=None, drop_na=True):
        if data is None:
            self.data = pd.DataFrame({'x':x,'y':y})
            x, y = 'x', 'y'
        else:
            self.data = data
        
        self.x_label, self.y_label = x, y

        if drop_na:
            self.data = self.data.loc[:, [x,y]].dropna()
        elif self.data.loc[:, [x,y]].isna().any(None):
            raise ValueError("Data contains NaN values and drop_na has been set to False.")

        self.m, self.b, self.r, self.p, self.e = scipy.stats.linregress(self.data[x], self.data[y])
        
        self.x_pred = np.array([self.data[x].min(), self.data[x].max()])
        self.y_pred = self.m * self.x_pred + self.b