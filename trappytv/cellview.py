import os
import numpy as np
import pandas as pd


class CellView:
    def __init__(
        self,
        datapath,
        xycols=("x_unrefined", "y_unrefined"),
        keymap=None,
        compute_speed=True,
        verbose=True,
        protected_cols=None,
    ):
        """
        keymap: dict with logical -> HDF key
            defaults:
                tracks   -> "df" (explicit default, with fallback)
                metadata -> ["metadata","meta","tracks/meta","/tracks/meta/"]
                xyr_df   -> "xyr_df"

            if any entry is None => feature disabled
        """

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

        # =========================================================
        # KEYMAP (explicit default: "df")
        # =========================================================
        default_keymap = {
            "tracks": "df",
            "metadata": ["metadata", "meta", "tracks/meta", "/tracks/meta/"],
            "xyr_df": "xyr_df",
        }
        self.keymap = {**default_keymap, **(keymap or {})}

        self._log("INIT", f"datapath={datapath}")
        self._log("PROTECTED", self.protected)

        # =========================================================
        # PATH HANDLING
        # =========================================================
        if datapath.endswith(".hd5"):
            self.scopeid = "Trappy-Scope"
            self.paths = {"tracks": datapath}
        else:
            self.scopeid = os.path.basename(datapath)[:2]
            pp = os.path.join(datapath, "postprocess")
            self.paths = {
                "tracks": os.path.join(pp, "merged_tracks.hd5"),
                "xyr_df": os.path.join(pp, "xyr.hd5"),
            }

            ff = os.path.join(pp, "first_frames.npy")
            if os.path.exists(ff):
                self.first_frames = np.load(ff)

        # =========================================================
        # INSPECT HDF KEYS
        # =========================================================
        self.hd5keys = []
        with pd.HDFStore(self.paths["tracks"]) as store:
            self.hd5keys = [k.lstrip("/") for k in store.keys()]

        self._log("HDF_KEYS", self.hd5keys)

        # =========================================================
        # RESOLVE MAIN KEY (explicit default + fallback)
        # =========================================================
        requested = self.keymap["tracks"]

        if requested is None:
            self._log("SKIP", "tracks disabled")
            self.main_key = None
        elif requested in self.hd5keys:
            self.main_key = requested
        else:
            self._log(
                "WARN",
                f"'{requested}' not found → fallback to 'tracks' if available",
            )
            if "tracks" in self.hd5keys:
                self.main_key = "tracks"
            else:
                self._log("ERROR", "no valid tracks key found")
                self.main_key = None

        self._log("MAIN_KEY", self.main_key)

        # =========================================================
        # LOAD DATA
        # =========================================================
        self.dfs = {}
        self._load_tracks()
        self._load_xyr()
        self._load_metadata()

        # =========================================================
        # SPEED
        # =========================================================
        if self.compute_speed_flag:
            self._compute_speed()

        ## Add Stuff
        self.stuff = {}

    # =========================================================
    # LOGGING
    # =========================================================
    def _log(self, stage, msg):
        if self.verbose:
            print(f"[{stage}] {msg}")

    # =========================================================
    # LOADERS
    # =========================================================
    def _load_tracks(self):
        if self.main_key is None:
            return
        try:
            self.dfs["tracks"] = pd.read_hdf(
                self.paths["tracks"], key=self.main_key
            )
            self._log("LOAD", "tracks loaded")
        except Exception as e:
            self._log("ERROR", f"tracks failed: {e}")

    def _load_xyr(self):
        key = self.keymap.get("xyr_df")
        if key is None:
            self._log("SKIP", "xyr_df disabled")
            return

        try:
            self.dfs["xyr_df"] = pd.read_hdf(self.paths["tracks"], key=key)
            self._log("LOAD", "xyr_df loaded")
        except Exception:
            self._log("WARN", "xyr_df missing")

    def _load_metadata(self):
        keys = self.keymap.get("metadata")
        if keys is None:
            self._log("SKIP", "metadata disabled")
            return

        if isinstance(keys, str):
            keys = [keys]

        self.metadata = None

        for k in keys:
            if k.strip("/") in self.hd5keys:
                try:
                    self.metadata = pd.read_hdf(self.paths["tracks"], key=k)
                    self._log("LOAD", f"metadata loaded ({k})")
                    return
                except Exception:
                    continue

        self._log("WARN", "metadata not found")

    # =========================================================
    # SPEED COMPUTATION
    # =========================================================
    def _compute_speed(self):
        df = self.dfs.get("tracks")
        if df is None:
            return

        ## Compute speed esentially recomputes speed.
        #if "speed" in df.columns:
        #    self._log("SPEED", "already present")
        #    return

        x, y = self.xycols

        if x not in df.columns or y not in df.columns:
            self._log("WARN", "speed skipped (missing xy columns)")
            return

        def compute(g):
            dx = np.gradient(g[x])
            dy = np.gradient(g[y])
            return np.hypot(dx, dy)

        if "scopeid" in df.columns:
            group_cols = [
                c for c in ["scopeid", "split", "particle"] if c in df.columns
            ]

            if group_cols:
                self._log("SPEED", f"grouped: {group_cols}")
                df["speed"] = (
                    df.groupby(group_cols, group_keys=False)
                    .apply(lambda g: pd.Series(compute(g), index=g.index))
                )
            else:
                self._log("SPEED", "particle present but no grouping cols")
                df["speed"] = compute(df)
        else:
            self._log("SPEED", "global")
            df["speed"] = compute(df)

    # =========================================================
    # API
    # =========================================================
    def add_columns(self, inputs, func, outputs, *args, inplace=True, **kwargs):
        """
        inplace: if True, func returns a new dataframe, otherwise it returns columns to be added to the dataframe.
        """
        df = self.dfs["tracks"]

        res = func(df, *args, inputs=inputs, outputs=outputs, **kwargs) if not inplace else func(df, *args, inputs=inputs, outputs=outputs, **kwargs)
        self._log("ADD_COL", f"{inputs} -> {list(set(res.columns) - set(df.columns)) if inplace else list(res.columns)}")

        if inplace:
            self.dfs["tracks"] = res.copy()
        else:
            self.dfs["tracks"] = self.dfs["tracks"].merge(
                                    res,
                                    left_index=True,
                                    right_index=True,
                                    how="left",
                                    suffixes=("", "_drop"))

            self.dfs["tracks"] = self.dfs["tracks"].drop(columns=[c for c in self.dfs["tracks"].columns if c.endswith("_drop")])

    def filter(self, func):
        df = self.dfs["tracks"]
        before = len(df)

        self.dfs["tracks"] = df[func(df)]

        after = len(self.dfs["tracks"])
        self._log("FILTER", f"{before} -> {after}")

    def keep_columns(self, cols):
        df = self.dfs["tracks"]
        cols = set(cols)

        bad = cols & self.protected
        if bad:
            self._log("WARN", f"protected cols explicitly passed: {bad}")

        keep = list((cols | self.protected) & set(df.columns))
        self.dfs["tracks"] = df[keep]

        self._log("TRIM", f"kept={keep}")

    # =========================================================
    def __call__(self):
        return self.dfs.get("tracks")