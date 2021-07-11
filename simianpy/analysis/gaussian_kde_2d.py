import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

# class from here: http://nbviewer.ipython.org/gist/tillahoffmann/f844bce2ec264c1c8cb5
#TODO: implement cupy support
#TODO: generalize to n-dimensional cases?
class GaussianKDE2D(object):
    """Representation of a kernel-density estimate using Gaussian kernels.
    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `GaussianKDE2D` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.
    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `GaussianKDE2D` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, shape (n, ), optional, default: None
        An array of weights, of the same shape as `x`.  Each value in `x`
        only contributes its associated weight towards the bin count
        (instead of 1).
    norm : bool, optional, default: True
        pass
    Constructors
    ------------
    GaussianKDE2D.from_dataframe(data, x, y, weights, bw_method, norm)

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `GaussianKDE2D` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : float
        Effective sample size using Kish's approximation.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.
    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.
    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `GaussianKDE2D`
    uses a rule of thumb, the default is Scott's Rule.
    Scott's Rule [1]_, implemented as `scotts_factor`, is::
        n**(-1./(d+4)),
    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::
        (n * (d + 2) / 4.)**(-1. / (d + 4)).
    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.
    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    Examples
    --------
    Generate some random two-dimensional data:
    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = np.random.normal(size=n)
    >>>     m2 = np.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2
    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()
    Perform a kernel density estimate on the data:
    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)
    Plot the results:
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()
    """
    def __init__(self, dataset, bw_method=None, weights=None, norm=True):
        self.dataset = np.atleast_2d(dataset)
        self.weights = weights
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        if np.isnan(self.dataset).any():
            warnings.warn('dataset contains NaN or inf values. Converting to 0')
            self.dataset = np.nan_to_num(self.dataset)
        if np.isnan(self.weights).any():
            warnings.warn('weights contains NaN or inf values. Converting to 0')
            self.weights = np.nan_to_num(self.weights)
        self.kernel = scipy.stats.gaussian_kde(self.dataset, bw_method, self.weights)
        # self.d, self.n = self.dataset.shape

        # if weights is not None:
        #     self.weights = weights / np.sum(weights)
        # else:
        #     self.weights = np.ones(self.n) / self.n

        # if norm:
        #     self.norm = 1
        # else:
        #     self.norm = weights.sum()

        # # Compute the effective sample size
        # # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        # self.neff = 1.0 / np.sum(self.weights ** 2)

        # self.set_bandwidth(bw_method=bw_method)
    
    @classmethod
    def from_dataframe(cls, data, x, y, weights=None, bw_method=None, norm=True):
        x, y, weights = data[x].values, data[y].values, weights if weights is None else data[weights].values
        return cls.from_arrays(x, y, weights, bw_method, norm)
    
    @classmethod
    def from_arrays(cls, x, y, weights=None, bw_method=None, norm=True):
        xy = np.stack([x,y])
        return GaussianKDE2D(xy, bw_method, weights, norm=norm)

    def evaluate(self, points=None, range=None, xticks=None, yticks=None, resolution=100, return_series=True, return_points=False):
        if points is None:
            x, y = self.dataset
            if range is not None:
                (xmin, xmax), (ymin, ymax) = range
                xticks = np.linspace(xmin, xmax, resolution)
                yticks = np.linspace(ymin, ymax, resolution)
            if xticks is None:
                xticks = np.linspace(x.min(), x.max(), resolution)
            if yticks is None:
                yticks = np.linspace(y.min(), y.max(), resolution)

            xx, yy = np.meshgrid(xticks, yticks)
            points = np.array([np.ravel(xx), np.ravel(yy)])
        result = self.kernel(points)
        result = pd.Series(result, index=pd.MultiIndex.from_arrays(points, names=['x','y']), name='density')
        if return_points:
            return result, points
        else:
            return result

    __call__ = evaluate