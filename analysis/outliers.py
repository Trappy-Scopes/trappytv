from turtle import pos
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from copy import deepcopy

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
    dfc.loc[now_filled, "postprocess"] = "filled"

    return dfc
    
def detect_outliers_local(
    df: pd.DataFrame,
    xcol: str = "xf",
    ycol: str = "yf",
    ep_col: str = "ep_unrefined",
    postprocess_label: str = "postprocess",
    k: float = 3.0,
    no_tbins: int = 500,
    ep_threshold: float = 0.45,
    anchor_after_weight: float = 0.65,
):
    """
    Detect large-displacement events in a track and classify them using
    localisation uncertainty (ep) at the two flanking frames.

    ep is preferred over signal intensity as a classifier because signal
    is susceptible to background non-uniformity, photobleaching, and
    illumination gradients. ep is computed locally per-particle and is
    blind to those field-wide effects.

    A physically motivated global threshold (ep_threshold=0.45) is used
    rather than a per-track statistical one, because ep_unrefined is
    stationary across acquisitions under stable imaging conditions.

    Displacement outliers are still detected via local MAD thresholding
    per time-bin, which adapts to local track dynamics.

    Event types
    -----------
    "large_disp"        : large displacement, ep at flanks is normal.
    "large_uncertainty" : large displacement corroborated by elevated ep
                          at one or both flanking frames.

    Parameters
    ----------
    df : pd.DataFrame
        Track dataframe, expected to contain xcol, ycol, ep_col, and
        optionally postprocess_label.
    xcol, ycol : str
        Refined position columns.
    ep_col : str
        Per-frame localisation uncertainty from trackpy.
    postprocess_label : str
        Column marking interpolated frames; their ep is not a real
        measurement and is excluded from classification.
    k : float
        MAD multiplier for displacement outlier detection.
    no_tbins : int
        Number of time-bins for local displacement thresholding.
    ep_threshold : float
        Physical upper bound on localisation uncertainty. Frames above
        this are considered poorly localised.
    anchor_after_weight : float
        Weight given to the destination frame (anchor_after) when
        computing the weighted ep. The source frame receives
        (1 - anchor_after_weight). Destination is weighted more because
        landing on the wrong particle is the more diagnostic failure mode.
    """

    x = df[xcol].values
    y = df[ycol].values
    ep = df[ep_col].values
    postprocess = (
        df[postprocess_label].values
        if postprocess_label in df.columns
        else np.full(len(x), None)
    )

    N = len(x)

    # ── displacement gradient ────────────────────────────────────────────────
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.hypot(dx, dy)

    # ── local displacement thresholds (MAD per chunk) ────────────────────────
    edges = np.linspace(0, N, no_tbins + 1, dtype=int)
    thresh_ds = np.zeros(N)

    for i in range(no_tbins):
        lo, hi = edges[i], edges[i + 1]
        chunk = ds[lo:hi]

        if len(chunk) < 5:
            med = np.nanmedian(ds)
            sigma = 1.4826 * np.nanmedian(np.abs(ds - med))
        else:
            med = np.nanmedian(chunk)
            sigma = 1.4826 * np.nanmedian(np.abs(chunk - med))

        thresh_ds[lo:hi] = med + k * sigma

    large_idx = np.where(ds > thresh_ds)[0]

    # ── helper: ep at a flank frame, NaN for invalid/interpolated ───────────
    def flank_ep(idx):
        """
        Return ep at idx, or NaN if the frame is out of bounds,
        interpolated, or has a missing ep value.
        Interpolated frames have synthetic positions and no real
        trackpy localisation, so their ep must not influence classification.
        """
        if idx < 0 or idx >= N:
            return np.nan
        if postprocess[idx] == "filled":
            return np.nan
        v = ep[idx]
        return np.nan if np.isnan(v) else v

    # ── cluster expansion + classification ───────────────────────────────────
    events = []
    visited = set()

    for idx in large_idx:
        if idx in visited:
            continue

        cluster = [idx]
        visited.add(idx)

        j = idx + 1
        while j < N and (ds[j] > thresh_ds[j] or j - cluster[-1] <= 1):
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

        anchor_before = max(cluster[0] - 1, 0)
        anchor_after  = min(cluster[-1] + 1, N - 1)
        bad_start     = cluster[0]
        bad_end       = cluster[-1]

        # ── weighted ep over valid flanks only ───────────────────────────────
        ep_before = flank_ep(anchor_before)
        ep_after = flank_ep(anchor_after)

        weights = []
        values = []

        if not np.isnan(ep_before):
            weights.append(1 - anchor_after_weight)
            values.append(ep_before)

        if not np.isnan(ep_after):
            weights.append(anchor_after_weight)
            values.append(ep_after)

        if values:
            # renormalise weights to sum to 1 if one flank was excluded
            w = np.array(weights)
            weighted_ep = float(np.dot(w / w.sum(), values))
            is_uncertain = weighted_ep > ep_threshold
        else:
            # both flanks are interpolated — no ep evidence, do not flag
            weighted_ep = np.nan
            is_uncertain = False

    

        events.append({
            "type"          : "large_uncertainty" if is_uncertain else "large_disp",
            "cluster_steps" : cluster,
            "bad_start"     : bad_start,
            "bad_end"       : bad_end,
            "n_bad_points"  : max(0, bad_end - bad_start + 1),
            "anchor_before" : anchor_before,
            "anchor_after"  : anchor_after,
            "skip_dist"     : float(np.hypot(
                                x[anchor_after] - x[anchor_before],
                                y[anchor_after] - y[anchor_before]
                             )),
            "excursion_dist": float(sum(ds[c] for c in cluster)),
            "max_step"      : float(ds[np.array(cluster)].max()),
            "ep_before"     : round(ep_before, 3) if not np.isnan(ep_before) else None,
            "ep_after"      : round(ep_after,  3) if not np.isnan(ep_after)  else None,
            "ep_weighted"   : round(weighted_ep, 3) if not np.isnan(weighted_ep) else None,
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
    ep_threshold=0.45,
    image=None,
    image_alpha=0.6,
    pad=5,
    figsize=(5, 10),
):
    import matplotlib.pyplot as plt

    df = cv.dfs["tracks"]
    dx = np.gradient(df[xcol])
    dy = np.gradient(df[ycol])
    ds = np.hypot(dx, dy)
    for i, e in enumerate(events):

        fig, axs = plt.subplots(
            nrows=4,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1, 1, 1]},
            sharex=False,
        )

        ax_traj, ax_sig, ax_ep, ax_size = axs

        lo = max(0, e["anchor_before"] - pad)
        hi = min(len(df), e["anchor_after"] + pad)

        xs = df[xcol].values[lo:hi]
        ys = df[ycol].values[lo:hi]
        
        sig = df[sigcol].values[lo:hi]
        ep = df[ep_col].values[lo:hi]
        size = df[size_col].values[lo:hi]
        ds_ = ds[lo:hi]

        postprocess = df[postprocess_label].values[lo:hi]

        n = len(xs)

        left_pad_end = min(pad, n)
        right_pad_start = max(n - pad, 0)
        mid_lo = left_pad_end
        mid_hi = right_pad_start

        # Classify this event's colour by type
        event_type = e.get("type", "unknown")
        region_color = "orange" if event_type == "large_uncertainty" else "red"

        # ── image underlay ──────────────────────────────────────────────────
        if image is not None:
            h, w = image.shape[:2]
            xmin = max(int(np.floor(np.nanmin(xs))), 0)
            xmax = min(int(np.ceil(np.nanmax(xs))), w)
            ymin = max(int(np.floor(np.nanmin(ys))), 0)
            ymax = min(int(np.ceil(np.nanmax(ys))), h)

            if xmax > xmin and ymax > ymin:
                cropped = image[ymin:ymax, xmin:xmax]
                ax_traj.imshow(
                    cropped,
                    extent=[xmin, xmax, ymin, ymax],
                    origin="lower",
                    alpha=image_alpha,
                    aspect="equal",
                    zorder=0,
                )
                ax_traj.set_xlim(xmin, xmax)
                ax_traj.set_ylim(ymin, ymax)

        # ── trajectory ─────────────────────────────────────────────────────
        ax_traj.plot(xs, ys, c="gray", alpha=0.3, zorder=2)
        ax_traj.scatter(xs, ys, c="black", s=10, alpha=0.6, zorder=3)
        ax_traj.scatter(
            xs[mid_lo:mid_hi],
            ys[mid_lo:mid_hi],
            c=region_color,
            s=20,
            zorder=4,
        )

        if "filled" in list(postprocess):
            mask = np.array(postprocess) == "filled"
            ax_traj.scatter(
                xs[mask],
                ys[mask],
                s=120,
                facecolors="none",
                edgecolors=region_color,
                linewidths=1.5,
                label="interpolated",
                zorder=5,
            )

        # Mark the two flanking frames explicitly
        for anchor_key in ["anchor_before", "anchor_after"]:
            abs_idx = e.get(anchor_key)
            if abs_idx is not None:
                rel_idx = abs_idx - lo
                if 0 <= rel_idx < n:
                    ax_traj.scatter(
                        xs[rel_idx],
                        ys[rel_idx],
                        marker="D",
                        s=60,
                        facecolors="none",
                        edgecolors="orange",
                        linewidths=1.5,
                        zorder=6,
                        label=anchor_key,
                    )

        max_ep_str = (
            f" | max flank ep={e['max_flank_ep']:.3f}"
            if "max_flank_ep" in e
            else ""
        )
        ax_traj.set_title(f"Segment {i} :: {event_type}{max_ep_str}")
        ax_traj.set_aspect("equal")

        # ── helper shading ──────────────────────────────────────────────────
        def shade(ax, color=region_color):
            ax.axvspan(0, left_pad_end, color="white", alpha=0.1)
            ax.axvspan(mid_lo, mid_hi, color=color, alpha=0.15)
            ax.axvspan(right_pad_start, n, color="white", alpha=0.1)

        # ── signal ─────────────────────────────────────────────────────────
        ax_sig.plot(sig, c="black", marker="o", linewidth=1)
        shade(ax_sig)
        ax_sig.set_ylabel("signal")

        # ── ep ─────────────────────────────────────────────────────────────
        ax_ep.plot(ep, c="blue", marker="o", linewidth=1)
        ax_ep.axhline(ep_threshold, color="orange", linestyle="--", linewidth=1,
                      label=f"threshold={ep_threshold}")
        ax_ep.legend(fontsize=7, loc="upper right")
        shade(ax_ep)
        ax_ep.set_ylabel("ep")

        # ── size ───────────────────────────────────────────────────────────
        ax_size.plot(ds_, c="green", marker="o", linewidth=1)
        shade(ax_size)
        ax_size.set_ylabel("ds")
        ax_size.set_xlabel("frame")

        plt.tight_layout()
        plt.show()

def mark_events_in_postprocess(
    df: pd.DataFrame,
    events: list = [],
    inputs: list = ["postprocess"],
    outputs: list = ["postprocess"]
):
    """
    Annotate dataframe rows using detected event segments.

    Parameters
    ----------
    df : pd.DataFrame
        Track dataframe.

    events : list of dict
        Output from detect_outliers_local().

    postprocess_col : str
        Column to write labels into.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with updated postprocess labels.
    """

    df = df.copy()

    postprocess_col = inputs[0]

    for event in events:
        start = event["bad_start"]
        end   = event["bad_end"]
        label = event["type"]

        if end >= start:
            df.loc[start:end, postprocess_col] = label

    return df


import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def interpolate_large_uncertainty(
    df_: pd.DataFrame,
    events=[],
    inputs=[],
    outputs=[],
    poly_order: int = 3,
):

    df = df_.copy()
    # initialize outputs
    df[outputs] = deepcopy(df[inputs])

    for event in events:
        if event["type"] != "large_uncertainty":
            continue

        start = event["bad_start"]
        end = event["bad_end"]

        anchor_before = event["anchor_before"]
        anchor_after = event["anchor_after"]

        cols_in = inputs
        cols_out = outputs

        # local window (safe copy)
        local = df.loc[anchor_before:anchor_after, cols_out].copy()

        # mask only event region inside local frame
        local.loc[start:end, [col for col in cols_out if col != "postprocess"]] = np.nan

        # interpolate per column
        #for c in cols_out:
        #    local[c] = local[c].interpolate(
        #        method="polynomial",
        #        order=poly_order if event["n_bad_points"] > poly_order else max(1, event["n_bad_points"]-1))
        #        # To handle small events.
        # write back repaired region
        df.loc[start:end, cols_out] = local.loc[start:end, cols_out].to_numpy()

    ## Interpolate the rest, i.e. missing.
    df[outputs[0]] = df[outputs[0]].interpolate(method="polynomial", order=poly_order)
    df[outputs[1]] = df[outputs[1]].interpolate(method="polynomial", order=poly_order)
    # optional metadata column (must be explicit, not inputs[2])
    #df.loc[start:end, "postprocess"] = "filtered"

    return df