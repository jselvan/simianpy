from scipy.stats import ttest_ind
import xarray as xr

def xr_ttest_ind(a, b, dim):
    remaining_dims = set(a.dims) - set((dim,))
    t, p = ttest_ind(a, b, axis=a.get_axis_num(dim))
    stat = xr.Dataset(
        {
            't': (remaining_dims, t),
            'p': (remaining_dims, p)
        },
        coords={k: v for k, v in a.coords.items() if k in remaining_dims}
    )
    return stat

def xr_ttest_ind_by_var(data, var, a, b, dim):
    a = data.query({dim: f"{var}=='{a}'"})
    b = data.query({dim: f"{var}=='{b}'"})
    return xr_ttest_ind(a, b, dim=dim)