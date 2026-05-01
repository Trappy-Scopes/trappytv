from typing import Callable, Optional, Sequence, Tuple

from bokeh.io import show
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.transform import transform

from .trappytv import TrappyTV


HoverBuilder = Callable[[Sequence[str]], Sequence[Tuple[str, str]]]


class TrappyTV2(TrappyTV):
    """
    trappytv viewer with explicit view modes:
    - view_split   (single split)
    - view_all     (all data with optional sampling)
    - view_ensamble (ensemble trajectory mode; API spelling preserved)

    Hover tooltips are user-configurable through `hover_builder`.

    Arguements:
    - default_xycols: "Default columns for xy trajectory rendering.
    - filtered_columns: Generate additional lines and checkboxes to view in `view_split` mode.
    """

    def __init__(
        self,
        cell,
        width=1000,
        height=1000,
        figs_width=140 * 5,
        figs_height=None,
        default_xycols=["x_unrefined", "y_unrefined"],
        filtered_columns={"filtered": ["xf", "yf"], "denoised": ["xf_7Hz", "yf_7Hz"], "anti-aliased": ["xf_3Hz", "yf_3Hz"]},
        hover_builder: Optional[HoverBuilder] = None,
        hover_columns: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            cell=cell,
            width=width,
            height=height,
            figs_width=figs_width,
            figs_height=figs_height,
            filtered_columns=filtered_columns
        )

        
        self.default_xycols = default_xycols
        self.filtered_columns = filtered_columns

        self.hover_builder: HoverBuilder = hover_builder or self._default_hover_builder
        self.hover_columns = tuple(hover_columns) if hover_columns is not None else None
        self._apply_hover_tooltips(self.hover_columns)

    def set_hover_builder(self, hover_builder: HoverBuilder, refresh=True):
        """Set/replace the hover tooltip builder callback."""
        self.hover_builder = hover_builder
        if refresh:
            self._apply_hover_tooltips(self.hover_columns)

    def set_hover_columns(self, hover_columns: Sequence[str], refresh=True):
        """Set/replace columns used for auto hover tooltip generation."""
        self.hover_columns = tuple(hover_columns)
        if refresh:
            self._apply_hover_tooltips(self.hover_columns)

    def _auto_hover_builder(self, hover_columns: Sequence[str]) -> Sequence[Tuple[str, str]]:
        """Automatically build tooltips from provided column names."""
        valid = [col for col in hover_columns if col in self.columns]
        tooltips = []
        for col in valid:
            if col == "dt":
                tooltips.append((col, "@dt{%F %T}"))
            else:
                tooltips.append((col, f"@{col}"))
        return tooltips

    def _default_hover_builder(self, hover_columns: Sequence[str]) -> Sequence[Tuple[str, str]]:
        """Default hover tooltip config; can be replaced by user callback."""
        if hover_columns:
            return self._auto_hover_builder(hover_columns)

        tooltips = []
        if "gframe_" in self.columns:
            tooltips.append(("gframe_", "@gframe_"))
        elif "gframe" in self.columns:
            tooltips.append(("gframe", "@gframe"))
        if "split" in self.columns:
            tooltips.append(("split", "@split"))
        if "x" in self.columns and "y" in self.columns:
            tooltips.append(("(X, Y)", "(@x, @y)"))
        if "frame" in self.columns:
            tooltips.append(("frame", "@frame"))
        if "dt" in self.columns:
            tooltips.append(("dt", "@dt{%F %T}"))
        return tooltips

    def _apply_hover_tooltips(self, hover_columns: Optional[Sequence[str]] = None):
        """Apply hover tooltips from user-provided builder."""
        columns = tuple(hover_columns) if hover_columns is not None else (self.hover_columns or ())
        tips = list(self.hover_builder(columns)) if callable(self.hover_builder) else []
        has_dt = any(field == "dt" for field, _ in tips)
        for tool in self.fig.tools:
            if isinstance(tool, HoverTool):
                if self.scatter in tool.renderers:
                    tool.tooltips = tips
                    if has_dt:
                        tool.formatters = {"@dt": "datetime"}
                    break

    def _clear_main_renderers(self):
        for renderer_name in ("scatter", "line"):
            renderer = getattr(self, renderer_name, None)
            if renderer is not None and renderer in self.fig.renderers:
                self.fig.renderers.remove(renderer)

    def _render_main_glyphs(
        self,
        xycols,
        line_alpha=0.4,
        line_color="gray",
        *,
        line_source=None,
        scatter_source=None,
        scatter_color="blue",
    ):
        if line_source is None:
            self.line = self.fig.line(
                self.df[xycols[0]],
                self.df[xycols[1]],
                color=line_color,
                alpha=line_alpha,
                line_width=2,
                legend_label="Path",
            )
        else:
            self.line = self.fig.line(
                x=xycols[0],
                y=xycols[1],
                source=line_source,
                color=line_color,
                alpha=line_alpha,
                line_width=2,
                legend_label="Path",
            )
        self.scatter = self.fig.scatter(
            x=xycols[0],
            y=xycols[1],
            source=scatter_source or self.source,
            size=1,
            alpha=0.6,
            color=scatter_color,
            legend_label="Points",
            nonselection_alpha=0.0,
        )

    def _finalize_and_show(
        self,
        frame_col="gframe_",
        title=None,
        disable_split_slider=False,
        hover_columns: Optional[Sequence[str]] = None,
    ):
        self.__render_interaction__(frame_col=frame_col)
        self._apply_hover_tooltips(hover_columns=hover_columns)
        if disable_split_slider and hasattr(self, "split_slider"):
            self.split_slider.disabled = True
        if title is not None:
            self.fig.title.text = title
        show(self.layout)

    def _prepare_ensemble_source(
        self,
        xycols=("x", "y"),
        smooth_window=25,
        exclude_open=False,
    ):
        import numpy as np
        import pandas as pd
        from bokeh.palettes import Category20, Turbo256

        # Keep baseline df unchanged; operate on a local view dataframe.
        mode_df = self.df
        if exclude_open and "trap_open" in mode_df.columns:
            mode_df = mode_df[mode_df["trap_open"] != True]

        max_pairs = (
            mode_df[["split", "particle", "scopeid"]]
            .drop_duplicates()
            .groupby("split", observed=True)
            .size()
            .max()
        )
        n = int(max_pairs)
        if n <= 2:
            palette = ["#1f77b4", "#ff7f0e"]
        elif n <= 20:
            palette = list(Category20[max(3, n)])
        else:
            step = max(1, 256 // n)
            palette = [Turbo256[i * step] for i in range(n)]

        combos = mode_df[["split", "particle", "scopeid"]].drop_duplicates()
        combos["_rank"] = combos.groupby("split", observed=True).cumcount()
        combos["color"] = combos["_rank"].apply(lambda r: palette[r % len(palette)])
        combos = combos.drop(columns="_rank")
        mode_df = mode_df.merge(combos, on=["split", "particle", "scopeid"], how="left")

        frame_col = "frame" if "frame" in mode_df.columns else "gframe_"
        all_speed = {}
        for split_val, grp in mode_df.groupby("split", observed=True):
            xs, ys, colors = [], [], []
            for (_, _), pgrp in grp.groupby(["particle", "scopeid"], sort=False, observed=True):
                pgrp = pgrp.sort_values(frame_col)
                fx = pgrp[xycols[0]].to_numpy(dtype=float)
                fy = pgrp[xycols[1]].to_numpy(dtype=float)
                ft = pgrp[frame_col].to_numpy(dtype=float)

                dx = np.diff(fx)
                dy = np.diff(fy)
                dt = np.diff(ft)
                dt[dt == 0] = np.nan

                raw_speed = np.sqrt(dx ** 2 + dy ** 2) / dt
                raw_speed = np.concatenate([[np.nan], raw_speed])
                smoothed = (
                    pd.Series(raw_speed)
                    .rolling(window=smooth_window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )

                mask = ~np.isnan(smoothed)
                xs.append(ft[mask].tolist())
                ys.append(smoothed[mask].tolist())
                colors.append(pgrp["color"].iloc[0])

            all_speed[str(int(split_val))] = {"xs": xs, "ys": ys, "colors": colors}

        initial_split = int(mode_df["split"].min())
        initial = mode_df[mode_df["split"] == initial_split]
        init_speed = all_speed[str(initial_split)]

        return {
            "mode_df": mode_df,
            "frame_col": frame_col,
            "all_speed": all_speed,
            "initial_split": initial_split,
            "initial": initial,
            "init_speed": init_speed,
        }

    def view_split(
        self,
        split_no=0,
        xycols=("x_unrefined", "y_unrefined"),
        line_alpha=0.4,
        line_color="gray",
        hover_columns: Optional[Sequence[str]] = None,
    ):
        self._clear_main_renderers()
        self.split_no = split_no
        split_df = self.df[self.df.split == split_no]
        self.source.data = ColumnDataSource.from_df(split_df)

        self.color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=self.df["frame"].min(),
            high=self.df["frame"].max(),
        )
        self._render_main_glyphs(
            xycols=xycols,
            line_alpha=line_alpha,
            line_color=line_color,
            line_source=self.source,
            scatter_source=self.source,
            scatter_color=transform("frame", self.color_mapper),
        )
        self.render_sides_source()
        self._finalize_and_show(
            frame_col="frame" if "frame" in self.df.columns else "gframe_",
            title=f"trappytv :: Cell: {self.scopeid} :: Split view :: split:: {split_no}",
            disable_split_slider=True,
            hover_columns=hover_columns,
        )

    def view_all(
        self,
        sample=10,
        xycols=("x_unrefined", "y_unrefined"),
        line_alpha=0.4,
        line_color="gray",
        hover_columns: Optional[Sequence[str]] = None,
    ):
        self._clear_main_renderers()
        self.source.data = ColumnDataSource.from_df(self.df[::sample])

        self.color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=self.df["gframe_"].min(),
            high=self.df["gframe_"].max(),
        )
        self._render_main_glyphs(
            xycols=xycols,
            line_alpha=line_alpha,
            line_color=line_color,
            line_source=None,
            scatter_source=self.source,
            scatter_color=transform("gframe_", self.color_mapper),
        )
        self.render_sides_all_lines()
        self._finalize_and_show(
            frame_col="gframe_",
            title=f"trappytv :: Cell: {self.scopeid} :: Full-view – Sampling: #{sample}",
            disable_split_slider=True,
            hover_columns=hover_columns,
        )

    def view_ensamble(
        self,
        xycols=("x", "y"),
        line_alpha=0.4,
        line_color="gray",
        smooth_window=25,
        exclude_open=False,
        hover_columns: Optional[Sequence[str]] = None,
    ):
        self._clear_main_renderers()
        if self.fig.legend:
            self.fig.legend[0].items = []
        prepared = self._prepare_ensemble_source(
            xycols=xycols,
            smooth_window=smooth_window,
            exclude_open=exclude_open,
        )
        mode_df = prepared["mode_df"]
        frame_col = prepared["frame_col"]
        all_speed = prepared["all_speed"]
        initial_split = prepared["initial_split"]
        initial = prepared["initial"]
        init_speed = prepared["init_speed"]

        self.full_source = ColumnDataSource(mode_df)
        self.display_source = ColumnDataSource(initial)
        self.source = self.display_source

        self.speed_source = ColumnDataSource(
            data=dict(xs=init_speed["xs"], ys=init_speed["ys"], colors=init_speed["colors"])
        )

        self.line = self.fig.line(
            x=xycols[0],
            y=xycols[1],
            source=ColumnDataSource({xycols[0]: [], xycols[1]: []}),
            color=line_color,
            alpha=0.0 * line_alpha,
            line_width=0,
        )
        self.scatter = self.fig.scatter(
            x=xycols[0],
            y=xycols[1],
            source=self.display_source,
            size=0.10,
            alpha=0.6,
            color="color",
            nonselection_alpha=0.0,
        )

        self.fig2.renderers = []
        self.fig2.multi_line(
            xs="xs",
            ys="ys",
            source=self.speed_source,
            line_color="colors",
            line_width=1.5,
            alpha=0.8,
        )
        self.fig2.xaxis.axis_label = frame_col
        self.fig2.yaxis.axis_label = f"speed (rolling {smooth_window}fr)"
        self.fig2.title.text = f"Speed per trajectory (window={smooth_window})"

        _empty = ColumnDataSource({"x": [], "y": []})
        self.fig3.renderers = []
        self.fig3.scatter(x="x", y="y", source=_empty, alpha=0)
        self.fig4.renderers = []
        self.fig4.scatter(x="x", y="y", source=_empty, alpha=0)

        callback = CustomJS(
            args=dict(
                display_source=self.display_source,
                full_source=self.full_source,
                speed_source=self.speed_source,
                all_speed=all_speed,
                plot_title=self.fig.title,
            ),
            code="""
            const split_val = cb_obj.value;
            const split_key = String(split_val);
            const full = full_source.data;
            const nd = {};
            for (const col of Object.keys(full)) { nd[col] = []; }

            for (let i = 0; i < full['split'].length; i++) {
                if (full['split'][i] === split_val) {
                    for (const col of Object.keys(full)) {
                        nd[col].push(full[col][i]);
                    }
                }
            }
            display_source.data = nd;

            const sd = all_speed[split_key];
            if (sd !== undefined) {
                speed_source.data = { xs: sd['xs'], ys: sd['ys'], colors: sd['colors'] };
            }
            plot_title.text = 'Trajectories — split ' + split_val;
            """,
        )
        self.split_slider.js_on_change("value", callback)
        self._finalize_and_show(
            frame_col=frame_col,
            title=f"trappytv :: Cell: {self.scopeid} :: Ensemble view :: split:: {self.split_no}",
            hover_columns=hover_columns,
        )
