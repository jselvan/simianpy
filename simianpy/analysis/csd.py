import numpy as np 

def CSD(z, data, slices=None, sigma=-0.3):
    # 5 point approximation of second spatial derivative
    # Freeman and Nicholson 1975
    # z is in microns
    # if slices, chunk up data
    # sigma = -0.3 (Petterson et al 2006)
    # applies to regularly spaced 1D data
    if slices is not None:
        z = np.stack([z[l:r].mean(axis=0) for l,r in zip(slices[:-1],slices[1:])])
        data = np.stack([data[l:r, :].mean(axis=0) for l,r in zip(slices[:-1],slices[1:])])

    diff = np.diff(z)
    if np.unique(diff).size == 1:
        h, = np.unique(diff)
    else:
        raise ValueError

    data_ = sigma * ( (data[4:]-data[2:-2]+data[:-4]) / 4*(h**2) )
    z_ = z[2:-2]
    return z_, data_