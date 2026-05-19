import os
import numpy as np
import pandas as pd
import h5py


class CellView:
    """
    Reads a Trappy-Scope cell-set HDF5 file as defined in the Cell Atlas spec.

    A cell set contains up to five components (§3):
        tracks    – trajectory DataFrame              (HDF key: "df")
        xyr       – trap-centre trajectory            (HDF key: "xyr" or "xyr/{eid}")
        metadata  – microscope / sample attrs         (h5py attrs on "metadata" or "metadata/{eid}")
        fov       – first-frame snapshot              (HDF dataset "fov" or "fov/{eid}")
        counts    – cell counts per split [optional]  (HDF key: "counts")

    Single-cell vs. ensemble is auto-detected via the §5.1 discriminant:
        ensemble ⟺ hf["metadata"] is an h5py.Group with no top-level attrs.
    """

    def __init__(
        self,
        datapath,
        xycols=("x_unrefined", "y_unrefined"),   # §5.2 column names
        compute_speed=True,
        verbose=True,
        protected_cols=None,
    ):
        self.xycols = xycols
        self.compute_speed_flag = compute_speed
        self.verbose = verbose
        self.stuff = {}

        # =========================================================
        # PROTECTED COLUMNS
        # =========================================================
        base_protected = {"frame", "dt", "split", "scopeid", "particle"}
        self.protected = (
            base_protected
            if protected_cols is None
            else base_protected | set(protected_cols)
        )

        self._log("INIT", f"datapath={datapath}")
        self._log("PROTECTED", self.protected)

        # =========================================================
        # PATH RESOLUTION
        # Single HDF5 file  -> use directly.
        # Directory         -> <dir>/postprocess/merged_tracks.hd5
        # =========================================================
        if datapath.endswith(".hd5") or datapath.endswith(".h5"):
            self.hdf_path = datapath
            self.scopeid = "Trappy-Scope"
        else:
            self.scopeid = os.path.basename(datapath)[:2]
            pp = os.path.join(datapath, "postprocess")
            self.hdf_path = os.path.join(pp, "merged_tracks.hd5")

            ff = os.path.join(pp, "first_frames.npy")
            if os.path.exists(ff):
                self.first_frames = np.load(ff)

        self._log("HDF_PATH", self.hdf_path)

        # =========================================================
        # INSPECT PANDAS-VISIBLE HDF KEYS
        # =========================================================
        with pd.HDFStore(self.hdf_path, "r") as store:
            self.hd5keys = {k.lstrip("/") for k in store.keys()}

        self._log("HDF_KEYS", self.hd5keys)

        # =========================================================
        # DETECT ENSEMBLE  (§5.1 discriminant)
        # =========================================================
        self.ensemble = self._detect_ensemble()
        self._log("ENSEMBLE", self.ensemble)

        # =========================================================
        # LOAD ALL COMPONENTS
        # =========================================================
        self.dfs = {}
        self.metadata = None
        self.fov = None
        self.counts = None

        self._load_tracks()
        self._load_xyr()
        self._load_metadata()
        self._load_fov()
        self._load_counts()

        # =========================================================
        # SPEED
        # =========================================================
        if self.compute_speed_flag:
            self._compute_speed()

    # =========================================================
    # LOGGING
    # =========================================================
    def _log(self, stage, msg):
        if self.verbose:
            print(f"[{stage}] {msg}")

    # =========================================================
    # ENSEMBLE DETECTION  (§5.1)
    # =========================================================
    def _detect_ensemble(self):
        """
        ensemble ⟺ hf["metadata"] is an h5py.Group AND has no top-level attrs.
        Returns False if "metadata" is absent (can't determine; assume single-cell).
        """
        try:
            with h5py.File(self.hdf_path, "r") as hf:
                if "metadata" not in hf:
                    self._log("WARN", "no 'metadata' node; assuming single-cell")
                    return False
                return (
                    isinstance(hf["metadata"], h5py.Group)
                    and not hf["metadata"].attrs.keys()
                )
        except Exception as e:
            self._log("WARN", f"ensemble detection failed: {e}")
            return False

    # =========================================================
    # LOADERS
    # =========================================================
    def _load_tracks(self):
        """HDF key is always 'df' (§5.1 line 4). No fallback — a missing key
        means the file is malformed."""
        try:
            self.dfs["tracks"] = pd.read_hdf(self.hdf_path, key="tracks")
            self._log("LOAD", f"tracks loaded ({len(self.dfs['tracks'])} rows)")
        except Exception as e:
            self._log("ERROR", f"tracks failed: {e}")

    def _load_xyr(self):
        """
        §5.1:
          single-cell  -> pd.read_hdf(path, key="xyr")          -> DataFrame
          ensemble     -> {eid: pd.read_hdf(path, key="xyr/eid")} -> dict[str, DataFrame]
        """
        try:
            with h5py.File(self.hdf_path, "r") as hf:
                if "xyr" not in hf:
                    self._log("WARN", "xyr not found")
                    return

                if self.ensemble:
                    eids = list(hf["xyr"].keys())
                    self.dfs["xyr"] = {
                        eid: pd.read_hdf(self.hdf_path, key=f"xyr/{eid}")
                        for eid in eids
                    }
                    self._log("LOAD", f"xyr loaded for eids: {eids}")
                else:
                    self.dfs["xyr"] = pd.read_hdf(self.hdf_path, key="xyr")
                    self._log("LOAD", "xyr loaded")
        except Exception as e:
            self._log("WARN", f"xyr failed: {e}")

    def _load_metadata(self):
    try:
        with pd.HDFStore(self.hdf_path, "r") as store:
            if self.ensemble:
                eids = [k.split("/")[-1] for k in store.keys() if k.startswith("/metadata/")]
                self.metadata = {
                    eid: store[f"/metadata/{eid}"].iloc[0].to_dict()
                    for eid in eids
                }
            else:
                self.metadata = store["/metadata"].iloc[0].to_dict()

    except Exception as e:
        self._log("WARN", f"metadata failed: {e}")


    def _load_fov(self):
        """
        single-cell  -> np.ndarray
        ensemble     -> {eid: np.ndarray}
        """
        try:
            with h5py.File(self.hdf_path, "r") as hf:
                if "fov" not in hf:
                    self._log("WARN", "fov not found")
                    return

                fov_node = hf["fov"]

                if self.ensemble:
                    if not isinstance(fov_node, h5py.Group):
                        raise TypeError("'fov' should be a group in ensemble mode")

                    self.fov = {
                        eid: fov_node[eid][()]
                        for eid in fov_node.keys()
                        if isinstance(fov_node[eid], h5py.Dataset)
                    }
                    self._log("LOAD", f"fov loaded for eids: {list(self.fov.keys())}")

                else:
                    if isinstance(fov_node, h5py.Dataset):
                        self.fov = fov_node[()]
                    elif isinstance(fov_node, h5py.Group):
                        # if single image stored inside group, take first dataset
                        first_key = next(iter(fov_node.keys()))
                        self.fov = fov_node[first_key][()]
                    else:
                        raise TypeError("Unsupported fov node type")

                    self._log("LOAD", f"fov loaded (shape={self.fov.shape})")

        except Exception as e:
            self._log("WARN", f"fov failed: {e}")

    def _load_counts(self):
        """
        §3 / §5.1: optional; single DataFrame regardless of ensemble type.
        Checks both the h5py tree and the pandas store (covers both storage styles).
        """
        try:
            with h5py.File(self.hdf_path, "r") as hf:
                in_h5 = "counts" in hf
            if in_h5 or "counts" in self.hd5keys:
                self.counts = pd.read_hdf(self.hdf_path, key="counts")
                self._log("LOAD", f"counts loaded ({len(self.counts)} rows)")
            else:
                self._log("SKIP", "counts not present")
        except Exception as e:
            self._log("WARN", f"counts failed: {e}")

    # =========================================================
    # SPEED COMPUTATION
    # =========================================================
    def _compute_speed(self):
        df = self.dfs.get("tracks")
        if df is None:
            return

        x, y = self.xycols
        if x not in df.columns or y not in df.columns:
            self._log("WARN", f"speed skipped (missing xy columns: {x!r}, {y!r})")
            return

        def compute(g):
            dx = np.gradient(g[x].values)
            dy = np.gradient(g[y].values)
            return np.hypot(dx, dy)

        # §2: single-cell groups by [split, particle]; ensemble adds scopeid (and eid).
        group_cols = [c for c in ["scopeid", "eid", "split", "particle"] if c in df.columns]

        if group_cols:
            self._log("SPEED", f"grouped by: {group_cols}")
            df["speed"] = df.groupby(group_cols, group_keys=False).apply(
                lambda g: pd.Series(compute(g), index=g.index)
            )
        else:
            self._log("SPEED", "global (no grouping columns found)")
            df["speed"] = compute(df)

        self.dfs["tracks"] = df

    # =========================================================
    # API
    # =========================================================
    def add_columns(self, inputs, func, outputs, *args, inplace=True, **kwargs):
        """
        Apply *func* to the tracks DataFrame and store the result.

        inplace=True  – func returns a complete replacement DataFrame.
        inplace=False – func returns only the new columns; they are merged in.
        """
        df = self.dfs["tracks"]
        res = func(df, *args, inputs=inputs, outputs=outputs, **kwargs)

        if inplace:
            new_cols = list(set(res.columns) - set(df.columns))
            self._log("ADD_COL", f"{inputs} -> {new_cols}")
            self.dfs["tracks"] = res.copy()
        else:
            self._log("ADD_COL", f"{inputs} -> {list(res.columns)}")
            merged = df.merge(
                res, left_index=True, right_index=True, how="left", suffixes=("", "_drop")
            )
            self.dfs["tracks"] = merged.drop(
                columns=[c for c in merged.columns if c.endswith("_drop")]
            )

    def filter(self, func):
        df = self.dfs["tracks"]
        before = len(df)
        self.dfs["tracks"] = df[func(df)]
        self._log("FILTER", f"{before} -> {len(self.dfs['tracks'])}")

    def keep_columns(self, cols):
        df = self.dfs["tracks"]
        cols = set(cols)

        bad = cols & self.protected
        if bad:
            self._log("WARN", f"protected cols explicitly passed: {bad}")

        keep = list((cols | self.protected) & set(df.columns))
        self.dfs["tracks"] = df[keep]
        self._log("TRIM", f"kept={keep}")

    def __call__(self):
        return self.dfs.get("tracks")
