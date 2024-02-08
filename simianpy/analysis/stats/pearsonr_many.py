from scipy.special import betainc
import numpy as np
def pearsonr_many(x, ys):
    x_mean = x.mean()
    y_means = ys.mean(axis=1)

    xm, yms = x - x_mean, ys - y_means[:, None]
    
    r = yms @ xm / np.sqrt(xm @ xm * (yms * yms).sum(axis=1))
    r = r.clip(-1, 1)

    prob = betainc(
        len(x) / 2 - 1,
        0.5,
        1 / (1 + r * r / (1 - r * r))
    )

    return r, prob