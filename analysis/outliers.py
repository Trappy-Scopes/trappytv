from turtle import pos
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd

def drop_edge_nans_and_interpolate(
    df:pd.DataFrame, inputs=["x_unrefined", "y_unrefined"],
    outputs=["xf", "yf"],
    max_length=5, poly_order=3):
    """
    1. Drop leading and trailing NANs and resets the indices.
    2. For each xycol, fill only nan gaps <= max_length using polynomial interpolation of given order (default 3). 
       Gaps longer than max_length are left as NaN.
    """

    dfc = df.copy()

    # 1. Drop leading and trailing NANs and reset indices.
    starts = [dfc[col].first_valid_index() for col in inputs]
    ends   = [dfc[col].last_valid_index()  for col in inputs]
    if any(v is None for v in starts + ends):
        return dfc.iloc[0:0]
    dfc = dfc.loc[max(starts):min(ends)].reset_index(drop=True)

    # 2. Copy xycols into new_xycols, mark NaNs as "missing".
    dfc[outputs] = dfc[inputs].values
    was_nan = dfc[outputs].isna().any(axis=1)
    dfc["postprocess"] = np.where(was_nan, "missing", None)

    # 3. Interpolate new_xycols only.
    dfc[outputs] = dfc[outputs].interpolate(
        limit=max_length, method='polynomial', order=poly_order,
        limit_direction="both", limit_area="inside"
    )

    # 4. Rows that were NaN but are now filled → "interpolated".
    now_filled = was_nan & dfc[outputs].notna().all(axis=1)
    dfc.loc[now_filled, "postprocess"] = "interpolated"

    return dfc
    
def detect_outliers_local(
    df,
    xcol="xf",
    ycol="yf",
    sigcol="signal",
    k: float = 3.0,
    no_tbins: int = 500,
):

    x = df[xcol].values
    y = df[ycol].values
    signal = df[sigcol].values

    N = len(x)

    # --- gradient + hypot ---
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.hypot(dx, dy)

    # --- local thresholds via chunking ---
    edges = np.linspace(0, N, no_tbins + 1, dtype=int)
    thresh_ds = np.zeros(N)

    for i in range(no_tbins):
        lo, hi = edges[i], edges[i + 1]
        chunk = ds[lo:hi]

        if len(chunk) < 5:
            thresh_ds[lo:hi] = np.nanmedian(ds) + k * (1.4826 * np.nanmedian(np.abs(ds - np.nanmedian(ds))))
            continue

        med = np.nanmedian(chunk)
        mad = np.nanmedian(np.abs(chunk - med))
        sigma = 1.4826 * mad

        thresh_ds[lo:hi] = med + k * sigma

    large_idx = np.where(ds > thresh_ds)[0]

    events = []
    visited = set()

    for idx in large_idx:
        if idx in visited:
            continue

        # --- cluster expansion ---
        cluster = [idx]
        visited.add(idx)

        j = idx + 1
        while j < len(ds) and (ds[j] > thresh_ds[j] or j - cluster[-1] <= 1):
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

        first_step = cluster[0]
        last_step = cluster[-1]

        anchor_before = first_step
        anchor_after = min(last_step + 1, N - 1)

        bad_start = first_step + 1
        bad_end = last_step
        n_bad = max(0, bad_end - bad_start + 1)

        skip_dist = np.hypot(
            x[anchor_after] - x[anchor_before],
            y[anchor_after] - y[anchor_before],
        )

        excursion_dist = sum(ds[c] for c in cluster)

        # --- signal dip ---
        bad_pts = np.arange(bad_start, bad_end + 1)

        if len(bad_pts) > 0:
            sig_segment = signal[bad_pts]
            sig_ref = np.nanmedian(signal[max(0, bad_start - 10):bad_start + 1])
            sig_drop = np.nanmean(sig_segment) < sig_ref
        else:
            sig_drop = False

        etype = "signal_dip" if sig_drop else "large_disp"

        events.append({
            "type": etype,
            "cluster_steps": cluster,
            "bad_start": bad_start,
            "bad_end": bad_end,
            "n_bad_points": n_bad,
            "anchor_before": anchor_before,
            "anchor_after": anchor_after,
            "skip_dist": skip_dist,
            "excursion_dist": excursion_dist,
            "max_step": ds[np.array(cluster)].max(),
        })

    return events


def plot_outlier_diagnostics_df(
    cv,
    events,
    xcol="xf",
    ycol="yf",
    sigcol="signal_unrefined",
    ep_col="ep_unrefined",
    size_col="size_unrefined",
    postprocess_label="postprocess",
    pad=5,
    figsize=(5, 10)
):
    import matplotlib.pyplot as plt
    import numpy as np

    df = cv.dfs["tracks"]

    for i, e in enumerate(events):

        fig, axs = plt.subplots(
            nrows=4,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1, 1, 1]},
            sharex=False
        )

        ax_traj, ax_sig, ax_ep, ax_size = axs

        lo = max(0, e["anchor_before"] - pad)
        hi = min(len(df), e["anchor_after"] + pad)

        xs = df[xcol].values[lo:hi]
        ys = df[ycol].values[lo:hi]
        sig = df[sigcol].values[lo:hi]
        ep = df[ep_col].values[lo:hi]
        size = df[size_col].values[lo:hi]
        postprocess = df[postprocess_label].values[lo:hi]

        n = len(xs)

        # ---------------- event region mask ----------------
        left_pad_end = min(pad, n)
        right_pad_start = max(n - pad, 0)

        mid_lo = left_pad_end
        mid_hi = right_pad_start

        # ---------------- trajectory ----------------
        ax_traj.plot(xs, ys, c="gray", alpha=0.3)

        # full markers (all points)
        ax_traj.scatter(xs, ys, c="black", s=10, alpha=0.6)

        # highlight event region
        ax_traj.scatter(xs[mid_lo:mid_hi], ys[mid_lo:mid_hi],
                        c="red", s=20)

        # interpolation markers
        if "interpolated" in list(postprocess):
            mask = np.array(postprocess) == "interpolated"
            ax_traj.scatter(
                xs[mask],
                ys[mask],
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                label="interpolated"
            )

        ax_traj.set_title(f"Segment: {i} :: type: {e['type']}")
        ax_traj.set_aspect("equal")

        # helper for consistent shading
        def shade(ax):
            ax.axvspan(0, left_pad_end, color="white", alpha=0.1)
            ax.axvspan(mid_lo, mid_hi, color="red", alpha=0.15)
            ax.axvspan(right_pad_start, n, color="white", alpha=0.1)

        # ---------------- signal ----------------
        ax_sig.plot(sig, c="black", marker="o", linewidth=1)
        shade(ax_sig)
        ax_sig.set_ylabel("signal")

        # ---------------- ep ----------------
        ax_ep.plot(ep, c="blue", marker="o", linewidth=1)
        shade(ax_ep)
        ax_ep.set_ylabel("ep")

        # ---------------- size ----------------
        ax_size.plot(size, c="green", marker="o", linewidth=1)
        shade(ax_size)
        ax_size.set_ylabel("size")
        ax_size.set_xlabel("frame")

        plt.tight_layout()
        plt.show()