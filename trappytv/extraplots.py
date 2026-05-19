import numpy as np

def subpix_hist(x):
    """Returns the subpixel histogram of x"""
    bias_x = x % 1
    return np.histogram(bias_x[~np.isnan(bias_x)], bins=10, density=True, range=[0.0, 1.0])

def residual_hist(x):
    """Return the residual histogram"""
    x = x[~np.isnan(x)]
    q75, q25 = np.percentile(x, [75 ,25])
    return np.histogram(x, bins=10, density=True, range=[q25, q75])