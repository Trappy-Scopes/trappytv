from hmac import new
from scipy.signal import savgol_coeffs, filtfilt
from copy import deepcopy

def savgol_filter_backandforth(x, window_length, polyorder, axis=-1):
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    b = savgol_coeffs(window_length, polyorder)

    y = filtfilt(b, [1.0], x, axis=axis, method='pad')
    return y



def apply_sg_bf_on_xy(
    df:pd.DataFrame, inputs=["xf", "yf"],
    outputs=["xf_sg", "yf_sg"],
    window_length=3, polyorder=3):
    """Apply a given Savgol filter configuration on the x and y coordinates.
       Applies both backwards and forwards filter to remove the phase lag introduced by a single operation.
    Returns: Only the new columns.   
    """
    new_df = deepcopy(df[[]])
    new_df[outputs[0]] = savgol_filter_backandforth(df[inputs[0]], window_length=window_length, polyorder=polyorder)
    new_df[outputs[1]] = savgol_filter_backandforth(df[inputs[1]], window_length=window_length, polyorder=polyorder)
    return new_df

import statsmodels.api as sm
def hodrick_prescott_on_xy(df:pd.DataFrame, inputs=["xf_3Hz", "yf_3Hz"],
    outputs=["xf_long", "yf_long"],
    lambda_smooth=1e3, include_cycles=False, cycle_outputs=["x_cycle", "y_cycle"],
    interpolate_if_necessary=False):
    """Apply the Hodrick-Prescott Filter to obtain the long-term component of the trajectory.
    if `include_cycle` is True, the cyclic part of the series is also included in the return.

    The filter fails with any NAN value present in the series. Therefore, NANs are checked before interpolation is performed with a filter.
    """
    new_df = deepcopy(df[[]])
    x = df[inputs[0]].copy()
    y = df[inputs[1]].copy()

    if sum(x.isna()) + sum(y.isna()) != 0:
        if not interpolate_if_necessary:
            raise Exception("NAN values in time-series. Filter will fail computation.")
        else:
            new_df["postprocess"] = df["postprocess"]
            x_nans = x.isna()
            y_nans = y.isna()
            new_df[x_nans]["postprocess"] = "hp_filter_interpolation"
            new_df[y_nans]["postprocess"] = "hp_filter_interpolation" ## Maybe redundant.
            x = x.interpolate(method='polynomial', order=3)
            y = y.interpolate(method='polynomial', order=3)
            print("[HP filter] Interpolation performed on gaps.")
    cyclex, new_df[outputs[0]] = sm.tsa.filters.hpfilter(x, lamb=lambda_smooth)
    cycley, new_df[outputs[1]] = sm.tsa.filters.hpfilter(y, lamb=lambda_smooth)

    if include_cycles:
        new_df[cycle_outputs[0]] = cyclex
        new_df[cycle_outputs[1]] = cycley
    return new_df
