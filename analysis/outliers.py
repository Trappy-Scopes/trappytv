
import numpy as np
import pandas as pd

def drop_edge_nans_and_interpolate(
    df:pd.DataFrame, xycols=["x_unrefined", "y_unrefined"],
    new_xycols=["xf", "yf"],
    max_length=5, poly_order=3):
    """
    1. Drop leading and trailing NANs and resets the indices.
    2. For each xycol, fill only nan gaps <= max_length using polynomial interpolation of given order (default 3). 
       Gaps longer than max_length are left as NaN.
    """

    dfc = df.copy()

    # 1. Drop leading and trailing NANs and reset indices.
    starts = [dfc[col].first_valid_index() for col in xycols]
    ends   = [dfc[col].last_valid_index()  for col in xycols]
    if any(v is None for v in starts + ends):
        return dfc.iloc[0:0]
    dfc = dfc.loc[max(starts):min(ends)].reset_index(drop=True)

    # 2. Copy xycols into new_xycols, mark NaNs as "missing".
    dfc[new_xycols] = dfc[xycols].values
    was_nan = dfc[new_xycols].isna().any(axis=1)
    dfc["postprocess"] = np.where(was_nan, "missing", None)

    # 3. Interpolate new_xycols only.
    dfc[new_xycols] = dfc[new_xycols].interpolate(
        limit=max_length, method='polynomial', order=poly_order,
        limit_direction="both", limit_area="inside"
    )

    # 4. Rows that were NaN but are now filled → "interpolated".
    now_filled = was_nan & dfc[new_xycols].notna().all(axis=1)
    dfc.loc[now_filled, "postprocess"] = "interpolated"

    return dfc
    


def detect_outliers_local(
    x: np.ndarray,
    y: np.ndarray,
    signal: np.ndarray,
    k: float = 5.0,
    k_signal: float = 3.0,       # signal uses a tighter threshold — it's less noisy
    window: int = 31,
    mistrack_ratio: float = 0.5,
) -> list[dict]:
    """
    Detect anomalous events in a trajectory using local adaptive MAD on both
    step size and signal intensity.

    Step-size outliers → mistrack_spike / mistrack_excursion / large_displacement
    Signal outliers with no step-size event → false_detection
    Signal is also used as corroborating evidence inside step-size clusters.

    Parameters
    ----------
    x, y       : trajectory coordinates
    signal     : trackpy signal (integrated brightness) at each frame
    k          : MAD threshold multiplier for step size
    k_signal   : MAD threshold multiplier for signal (typically lower than k)
    window     : rolling window size for local MAD (odd recommended)
    mistrack_ratio : return classification threshold
    """
    N = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    n_steps = len(ds)
    half = window // 2

    # ------------------------------------------------------------------ #
    # Helper: local MAD threshold for any 1-D array                       #
    # ------------------------------------------------------------------ #
    def local_thresh(arr: np.ndarray, multiplier: float) -> np.ndarray:
        out = np.empty(len(arr))
        for i in range(len(arr)):
            lo, hi = max(0, i - half), min(len(arr), i + half)
            nb = np.concatenate([arr[lo:i], arr[i+1:hi]])
            if len(nb) < 5:
                nb = arr
            med = np.nanmedian(nb)
            mad = np.nanmedian(np.abs(nb - med))
            out[i] = med + multiplier * 1.4826 * mad
        return out

    def local_median(arr: np.ndarray) -> np.ndarray:
        out = np.empty(len(arr))
        for i in range(len(arr)):
            lo, hi = max(0, i - half), min(len(arr), i + half)
            nb = np.concatenate([arr[lo:i], arr[i+1:hi]])
            out[i] = np.nanmedian(nb) if len(nb) >= 5 else np.nanmedian(arr)
        return out

    # ------------------------------------------------------------------ #
    # Step-size thresholds (per step, length N-1)                         #
    # ------------------------------------------------------------------ #
    thresh_ds = local_thresh(ds, k)
    large_idx = np.where(ds > thresh_ds)[0]

    # ------------------------------------------------------------------ #
    # Signal thresholds (per frame, length N)                             #
    # Signal can be anomalously LOW (noise peak, blink-out, false link)   #
    # or anomalously HIGH (debris, merged particle).                      #
    # Use two-sided: flag below local_median - k*sigma OR above thresh.   #
    # ------------------------------------------------------------------ #
    thresh_sig_hi = local_thresh(signal, k_signal)
    sig_local_med = local_median(signal)
    sig_mad       = local_thresh(np.abs(signal - sig_local_med), 1.0)  # local spread
    thresh_sig_lo = sig_local_med - k_signal * 1.4826 * sig_mad        # lower bound

    sig_anomalous = (signal > thresh_sig_hi) | (signal < thresh_sig_lo)

    # ------------------------------------------------------------------ #
    # Cluster expansion (same logic, now uses per-step thresh)            #
    # ------------------------------------------------------------------ #
    events  = []
    visited = set()

    for idx in large_idx:
        if idx in visited:
            continue

        cluster = [idx]
        visited.add(idx)

        j = idx + 1
        while j < n_steps and (ds[j] > thresh_ds[j] or j - cluster[-1] <= 1):
            if ds[j] > thresh_ds[j]:
                cluster.append(j)
                visited.add(j)
            j += 1

        j = idx - 1
        while j >= 0 and (ds[j] > thresh_ds[j] or cluster[0] - j <= 1):
            if ds[j] > thresh_ds[j]:
                cluster.insert(0, j)
                visited.add(j)
            j -= 1

        first_step    = cluster[0]
        last_step     = cluster[-1]
        anchor_before = first_step
        anchor_after  = min(last_step + 1, N - 1)
        bad_start     = first_step + 1
        bad_end       = last_step
        n_bad         = max(0, bad_end - bad_start + 1)

        skip_dist = np.sqrt(
            (x[anchor_after] - x[anchor_before])**2 +
            (y[anchor_after] - y[anchor_before])**2
        )
        excursion_dist = sum(ds[c] for c in cluster)

        # Local typical step just before the cluster
        pre_lo        = max(0, first_step - half)
        local_typical = np.nanmedian(ds[pre_lo:first_step])
        if np.isnan(local_typical) or local_typical == 0:
            local_typical = np.nanmedian(ds)

        # ---- Signal evidence inside the cluster ---- #
        # bad_start..bad_end are the point indices that are "inside" the event
        bad_pts         = np.arange(bad_start, bad_end + 1)
        bad_sig_anomaly = sig_anomalous[bad_pts].sum() if len(bad_pts) else 0
        bad_sig_values  = signal[bad_pts] if len(bad_pts) else np.array([])
        sig_drop        = bool(
            len(bad_pts) > 0 and
            np.nanmean(bad_sig_values) < sig_local_med[bad_pts].mean()
        )

        # ---- Classification ---- #
        ratio     = skip_dist / excursion_dist if excursion_dist > 0 else 1.0
        is_return = (ratio < mistrack_ratio) and (skip_dist < 3.0 * local_typical)

        if is_return and n_bad <= 5:
            etype = "mistrack_spike"
        elif is_return:
            etype = "mistrack_excursion"
        else:
            etype = "large_displacement"

        # Upgrade confidence: if signal is anomalous inside the cluster,
        # a large_displacement is more likely a mistrack to a false detection.
        if etype == "large_displacement" and bad_sig_anomaly > 0 and sig_drop:
            etype = "mistrack_excursion"   # revise: went to a dim/noise peak

        events.append({
            "type":             etype,
            "cluster_steps":    cluster,
            "bad_start":        bad_start,
            "bad_end":          bad_end,
            "n_bad_points":     n_bad,
            "anchor_before":    anchor_before,
            "anchor_after":     anchor_after,
            "skip_dist":        skip_dist,
            "excursion_dist":   excursion_dist,
            "ratio":            ratio,
            "local_typical":    local_typical,
            "max_step":         ds[np.array(cluster)].max(),
            "local_thresh":     thresh_ds[np.array(cluster)].max(),
            "n_sig_anomalous":  int(bad_sig_anomaly),
            "sig_drop":         sig_drop,
        })

    # ------------------------------------------------------------------ #
    # Signal-only false detections (no large step, just anomalous signal) #
    # These are missed entirely by the step-size test.                    #
    # ------------------------------------------------------------------ #
    step_bad_pts = set()
    for e in events:
        step_bad_pts.update(range(e["bad_start"], e["bad_end"] + 1))

    # Find isolated frames with anomalous signal not already in a cluster
    sig_only = np.where(sig_anomalous)[0]
    sig_visited = set()

    for pt in sig_only:
        if pt in step_bad_pts or pt in sig_visited:
            continue

        # Expand contiguous run of anomalous-signal frames
        run = [pt]
        sig_visited.add(pt)
        j = pt + 1
        while j < N and sig_anomalous[j] and j not in step_bad_pts:
            run.append(j)
            sig_visited.add(j)
            j += 1

        events.append({
            "type":            "false_detection",
            "bad_start":       run[0],
            "bad_end":         run[-1],
            "n_bad_points":    len(run),
            "anchor_before":   max(0, run[0] - 1),
            "anchor_after":    min(N - 1, run[-1] + 1),
            "sig_drop":        bool(signal[run].mean() < sig_local_med[run].mean()),
            "n_sig_anomalous": len(run),
            # step-size fields not applicable
            "cluster_steps":   [],
            "skip_dist":       0.0,
            "excursion_dist":  0.0,
            "ratio":           1.0,
            "max_step":        0.0,
        })

    events.sort(key=lambda e: e["bad_start"])
    return events

