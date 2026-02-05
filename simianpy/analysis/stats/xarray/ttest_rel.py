from scipy.stats import ttest_rel
import xarray as xr 

def xr_ttest_rel(a, b, dim, **kwargs):
    t, p = xr.apply_ufunc(
        ttest_rel,
        a, b,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        kwargs=kwargs
    )
    stat = t.to_dataset(name='t')
    stat['p'] = p
    return stat