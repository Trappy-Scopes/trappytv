"""
trappytv.py
───────────
TrappyTV: the public Bokeh visualisation widget for Trappy-Scopes trajectory data.

Inherits infrastructure from TrappyCore and adds:
- Configurable hover tooltip system (hover_builder / hover_columns)
- Configurable side-panel rendering (side_cols)
- Filtered-column overlays with checkbox visibility control (view_split only)
- Three view modes: view_split, view_all, view_ensemble

Usage
-----
    from trappytv import TrappyTV

    tv = TrappyTV(cell)
    tv.view_split(split_no=0)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from bokeh.palettes import Category20, Turbo256
from bokeh.transform import transform
from bokeh.models import LinearColorMapper
from .trappycore import TrappyCore, _DEFAULT_SIDE_COLS

HoverBuilder = Callable[[Sequence[str]], Sequence[Tuple[str, str]]]

# ── Overlay palette / markers ─────────────────────────────────────────────────
# Colours chosen to contrast clearly with the main Viridis scatter.
_OVERLAY_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
_OVERLAY_MARKERS = ["square", "triangle", "diamond", "hex", "star"]


class TrappyTV(TrappyCore):
    """
    Interactive Bokeh visualisation widget for Trappy-Scopes trajectory data.

    Parameters
    ----------
    cell : object
        Cell data container (see TrappyCore for details).
    width, height : int
        Main figure pixel dimensions. The right column width and side-panel
        heights are derived automatically (see TrappyCore for the formulae).
    default_xycols : sequence of str
        Default [x_col, y_col] used when xycols is not passed to a view method.
    filtered_columns : dict
        Checkbox-controlled overlay columns: {label: [x_col, y_col]}.
        In view_split, each entry renders a line + scatter overlay on the main
        figure; checkboxes toggle their visibility. Disabled in other view modes.
    side_cols : sequence of (str, str)
        Side-panel (data_column, display_label) pairs.
    hover_builder : callable, optional
        Function(columns) -> list of (label, spec) tooltip pairs.
    hover_columns : sequence of str, optional
        Column names passed to hover_builder.

    View modes
    ----------
    view_split(split_no, ...)    Single split with overlays; split slider is live.
    view_all(sample, ...)        Full trajectory; sampled interactive scatter.
    view_ensemble(...)           Multi-particle ensemble; per-split speed panel.
    """

    def __init__(
        self,
        cell,
        width: int = 1000,
        height: int = 1000,
        default_xycols: Sequence[str] = ("x_unrefined", "y_unrefined"),
        filtered_columns: dict = {
            "filtered":      ["xf",     "yf"    ],
            "denoised":      ["xf_11Hz", "yf_11Hz"],
            "anti-aliased":  ["xf_3Hz", "yf_3Hz"],
        },
        side_cols: Sequence[Tuple[str, str]] = _DEFAULT_SIDE_COLS,
        hover_builder: Optional[HoverBuilder] = None,
        hover_columns: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            cell=cell,
            width=width,
            height=height,
            default_xycols=default_xycols,
            filtered_columns=filtered_columns,
            side_cols=side_cols,
        )
        # default_xycols is now stored on self by TrappyCore.__init__ before
        # _build_widgets runs, so the X/Y selects pick up the correct defaults.
        self.hover_builder: HoverBuilder = hover_builder or self._default_hover_builder
        self.hover_columns: Optional[tuple] = (
            tuple(hover_columns) if hover_columns is not None else None
        )
        self._build_hover()

        # Overlay state -- populated by _render_filtered_overlays (view_split only),
        # cleared at the start of every view call via _clear_renderers override.
        self.overlay_renderers: list = []  # list of [line, scatter] pairs
        self.overlay_sources:   list = []  # one ColumnDataSource per valid overlay

    # ── Overlay clear / render / wire ─────────────────────────────────────────

    def _clear_renderers(self) -> None:
        """
        Override: clear overlays first, then delegate to the base implementation
        (which removes self.scatter / self.line and wipes the legend).
        """
        self._clear_overlays()
        super()._clear_renderers()

    def _clear_overlays(self) -> None:
        """Remove all overlay renderers from self.fig and reset overlay state."""
        for pair in self.overlay_renderers:
            for renderer in pair:
                if renderer in self.fig.renderers:
                    self.fig.renderers.remove(renderer)
        self.overlay_renderers = []
        self.overlay_sources   = []

    def _render_filtered_overlays(self, split_df) -> list:
        """
        Render one line + scatter overlay per valid entry in self.filtered_columns.

        'Valid' means both x_col and y_col exist in self.df. Invalid entries are
        silently skipped; the checkbox labels are updated to match.

        Parameters
        ----------
        split_df : DataFrame
            The initial split's rows; used to populate the overlay sources.

        Returns
        -------
        valid : list of (label, x_col, y_col)
            Only the entries that were actually rendered.
        """
        valid = [
            (label, cols[0], cols[1])
            for label, cols in self.filtered_columns.items()
            if cols[0] in self.columns and cols[1] in self.columns
        ]

        # Keep checkboxes in sync with valid overlays only.
        if self.checkboxes is not None:
            self.checkboxes.labels = [v[0] for v in valid]
            self.checkboxes.active = []   # overlays start hidden; raw_checkboxes are the only defaults

        self.overlay_renderers = []
        self.overlay_sources   = []

        # Read overlay appearance from the non-interactive sliders (set once at render
        # time; no JS callback — intentional to avoid expensive re-renders).
        ov_size  = self.overlay_size_slider.value  if self.overlay_size_slider  is not None else 3.0
        ov_alpha = self.overlay_alpha_slider.value if self.overlay_alpha_slider is not None else 0.5

        for i, (label, x_col, y_col) in enumerate(valid):
            color  = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
            marker = _OVERLAY_MARKERS[i % len(_OVERLAY_MARKERS)]

            source = ColumnDataSource({
                x_col: split_df[x_col].tolist(),
                y_col: split_df[y_col].tolist(),
            })
            self.overlay_sources.append(source)

            line = self.fig.line(
                x=x_col, y=y_col, source=source,
                color=color, alpha=ov_alpha, line_width=1,
                legend_label=label,
            )
            scatter = self.fig.scatter(
                x=x_col, y=y_col, source=source,
                color=color, marker=marker, size=ov_size, alpha=ov_alpha,
                legend_label=label,
                nonselection_alpha=0.0,
            )
            # Overlays are decorative -- don't interfere with main selection.
            # Lines are also non-interactive.
            line.nonselection_glyph    = None
            line.level                 = "underlay"
            line.hover_glyph           = None
            scatter.nonselection_glyph = None
            scatter.level              = "underlay"
            scatter.hover_glyph        = None
            self.overlay_renderers.append([line, scatter])

        return valid

    def _wire_checkboxes(self) -> None:
        """
        Wire the CheckboxButtonGroup to toggle overlay visibility.
        Called from _finalize when wire_checkboxes=True (view_split only).
        Clears any previously attached callbacks first.
        """
        if self.checkboxes is None or not self.overlay_renderers:
            return

        self.checkboxes.js_property_callbacks.pop("change:active", None)

        # Flat renderer list: [line0, scatter0, line1, scatter1, ...]
        flat_renderers = [r for pair in self.overlay_renderers for r in pair]

        cb = CustomJS(
            args=dict(renderers=flat_renderers),
            code="""
            const active = new Set(cb_obj.active);
            const n = Math.floor(renderers.length / 2);
            for (let i = 0; i < n; i++) {
                const vis          = active.has(i);
                renderers[2*i].visible   = vis;   // line
                renderers[2*i+1].visible = vis;   // scatter
            }
            """,
        )
        self.checkboxes.js_on_change("active", cb)

    # ── Hover system ──────────────────────────────────────────────────────────

    def set_hover_columns(self, hover_columns: Sequence[str], refresh: bool = True) -> None:
        """Replace the columns passed to hover_builder."""
        self.hover_columns = tuple(hover_columns)
        if refresh:
            self._apply_hover()

    def set_hover_builder(self, hover_builder: HoverBuilder, refresh: bool = True) -> None:
        """Replace the hover_builder callable."""
        self.hover_builder = hover_builder
        if refresh:
            self._apply_hover()

    def _default_hover_builder(self, hover_columns: Sequence[str]) -> Sequence[Tuple[str, str]]:
        """
        Build hover tooltip pairs from column names.
        If hover_columns is non-empty, only those columns are included.
        Otherwise a sensible default set is assembled from the dataframe.
        """
        if hover_columns:
            tips = []
            for col in hover_columns:
                if col not in self.columns:
                    continue
                tips.append((col, "@dt{%F %T}" if col == "dt" else f"@{col}"))
            return tips

        tips: list[Tuple[str, str]] = []
        for col in ("gframe_", "gframe"):
            if col in self.columns:
                tips.append((col, f"@{col}"))
                break
        if "split" in self.columns:
            tips.append(("split", "@split"))
        if "x" in self.columns and "y" in self.columns:
            tips.append(("(X, Y)", "(@x, @y)"))
        for col in ("frame", "particle", "scopeid"):
            if col in self.columns:
                tips.append((col, f"@{col}"))
        if "dt" in self.columns:
            tips.append(("dt", "@dt{%F %T}"))
        return tips

    def _build_hover(self) -> None:
        """Create a HoverTool and attach it to self.fig."""
        self.fig.tools = [t for t in self.fig.tools if not isinstance(t, HoverTool)]
        tips   = list(self.hover_builder(self.hover_columns or ()))
        has_dt = any(f == "dt" for f, _ in tips)
        self.hover = HoverTool(
            tooltips=tips,
            formatters={"@dt": "datetime"} if has_dt else {},
            renderers=[self.scatter],
        )
        self.fig.add_tools(self.hover)

    def _apply_hover(self, hover_columns: Optional[Sequence[str]] = None) -> None:
        """Update the existing HoverTool in place to point at the current self.scatter."""
        cols   = hover_columns if hover_columns is not None else (self.hover_columns or ())
        tips   = list(self.hover_builder(cols))
        has_dt = any(f == "dt" for f, _ in tips)
        for tool in self.fig.tools:
            if isinstance(tool, HoverTool):
                tool.tooltips   = tips
                tool.formatters = {"@dt": "datetime"} if has_dt else {}
                tool.renderers  = [self.scatter]
                return
        self._build_hover()

    # ── Side panel rendering ──────────────────────────────────────────────────

    def _render_sides(
        self,
        source: ColumnDataSource,
        frame_col: str = "gframe_",
        *,
        background_df=None,
    ) -> None:
        """
        Render all side figures from self.side_cols.

        Parameters
        ----------
        source : ColumnDataSource
            Interactive source linked to the main scatter.
        frame_col : str
            Column used as the x-axis on each side panel.
        background_df : DataFrame, optional
            When provided, draws a static grey background line spanning all data.
        """
        self.other_scatters = []

        for i, (fig, (y_col, label)) in enumerate(zip(self.side_figs, self.side_cols)):
            fig.renderers        = []
            fig.xaxis.axis_label = frame_col
            fig.yaxis.axis_label = y_col
            # Title intentionally blank (task 3); the y-axis label carries the info.

            if y_col not in self.df.columns:
                continue

            if background_df is not None and frame_col in background_df.columns:
                fig.line(
                    x=background_df[frame_col], y=background_df[y_col],
                    color="gray", alpha=0.2, line_width=2, level="underlay",
                )

            # Read size and alpha from per-panel sliders at render time.
            s_size  = self.side_size_sliders[i].value  if i < len(self.side_size_sliders)  else 3.0
            s_alpha = self.side_alpha_sliders[i].value if i < len(self.side_alpha_sliders) else 0.4

            scatter_color = (
                transform(frame_col, self.color_mapper)
                if self.color_mapper is not None
                else "steelblue"
            )
            scatter = fig.scatter(
                source=source,
                x=frame_col, y=y_col,
                color=scatter_color, size=s_size, alpha=s_alpha,
                nonselection_alpha=0.0,
            )
            self.other_scatters.append(scatter)

    # ── Shared finalise step ──────────────────────────────────────────────────

    def _finalize(
        self,
        *,
        title: Optional[str] = None,
        hover_columns: Optional[Sequence[str]] = None,
        disable_split_slider: bool = False,
        split_callback: Optional[CustomJS] = None,
        wire_checkboxes: bool = False,
        disable_checkboxes: bool = False,
    ) -> None:
        """
        Post-render wiring common to all view modes, then display.

        Order: title -> color_select -> hover -> style sliders -> xy selects
               -> split slider -> raw_checkboxes -> overlay checkboxes -> display.

        Parameters
        ----------
        wire_checkboxes : bool
            True only for view_split: enables and wires the overlay checkbox callbacks.
        disable_checkboxes : bool
            True for view_all and view_ensemble: disables the overlay checkbox widget.
            The raw_checkboxes (Raw line / Raw scatter) are always wired regardless.
        """
        if title is not None:
            self.fig.title.text = title

        if self.color_mapper is not None:
            self._wire_color_select(self.color_select.value)

        self._apply_hover(hover_columns)
        self._wire_style_sliders()
        self._wire_xy_selects()

        self.split_slider.js_property_callbacks.pop("change:value", None)
        self.split_slider.disabled = disable_split_slider
        if split_callback is not None:
            self.split_slider.js_on_change("value", split_callback)

        # Raw line / scatter toggles -- always wired for every view mode.
        # Reset both to visible first so the widget state matches reality on re-render.
        self.raw_checkboxes.active = [0, 1]
        self._wire_raw_checkboxes()

        # Side-panel column selects and style sliders -- always wired.
        # self.source, self.frame_col, and self.other_scatters are set by the
        # view method before _finalize is called.
        self._wire_side_selects()
        self._wire_side_style_sliders()

        # Overlay checkboxes -- only wired in view_split; disabled elsewhere.
        if self.checkboxes is not None:
            if disable_checkboxes:
                self.checkboxes.disabled = True
            else:
                self.checkboxes.disabled = False
                if wire_checkboxes:
                    self._wire_checkboxes()
                    # Wire overlay size/alpha sliders to the current overlay renderers.
                    # Safe to make interactive: only updates glyph visual properties,
                    # no data re-serialisation.
                    self._wire_overlay_sliders()

        # FOV checkbox — wired here so all view modes benefit.
        # _add_fov() is idempotent and is called before _finalize in every view
        # method; self.fov_renderer is None when no FOV was loaded.
        self._wire_trap_fov_checkboxes(fov_renderer=self.fov_renderer)

        self.display()

    # ── View modes ────────────────────────────────────────────────────────────

    def view_split(
        self,
        split_no: int = 0,
        xycols: Optional[Sequence[str]] = None,
        line_alpha: float = 0.4,
        line_color: str = "gray",
        hover_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Render a single split with filtered-column overlays.

        The split slider switches between splits live. Checkboxes toggle the
        visibility of each filtered-column overlay. Both update without a
        Python round-trip.

        Parameters
        ----------
        split_no : int
            Index into self.split_values for the initial split to render.
        xycols : [x_col, y_col], optional
            Overrides default_xycols.
        line_alpha, line_color : float, str
            Appearance of the main trajectory line.
        hover_columns : sequence of str, optional
            Overrides self.hover_columns for this call only.
        """
        xycols    = list(xycols or self.default_xycols)
        frame_col = "frame" if "frame" in self.columns else "gframe_"

        # _clear_renderers also calls _clear_overlays and wipes the legend.
        self._clear_renderers()
        self.split_no = split_no

        initial_split_val = self.split_values[split_no]
        initial_df = (
            self.df[self.df["split"] == initial_split_val]
            if "split" in self.df.columns
            else self.df
        )

        # full_source: all splits -- never mutated; used by the JS split callback.
        # display_source (self.source): the currently visible split; rewritten by JS.
        self.full_source = ColumnDataSource(self.df)
        self.source      = ColumnDataSource(initial_df)
        self.split_slider.value = split_no

        # ── Main trajectory glyphs ────────────────────────────────────────────
        color = transform(frame_col, self._make_color_mapper(frame_col))
        self._render_glyphs(
            xycols,
            line_source=self.source,
            scatter_source=self.source,
            scatter_color=color,
            line_alpha=line_alpha,
            line_color=line_color,
        )

        # ── Filtered-column overlays ──────────────────────────────────────────
        valid_overlays = self._render_filtered_overlays(initial_df)
        overlay_xcols  = [x for _, x, _ in valid_overlays]
        overlay_ycols  = [y for _, _, y in valid_overlays]

        # ── Side panels ───────────────────────────────────────────────────────
        self.frame_col = frame_col   # stored for _wire_side_selects in _finalize
        self._render_sides(self.source, frame_col=frame_col)

        # ── Histograms ────────────────────────────────────────────────────────
        # Pre-compute once in Python for all splits; JS will pick the right
        # entry on every slider change without a Python round-trip.
        all_hist_data = self._precompute_split_hists()
        initial_key   = str(int(initial_split_val))
        hist_sources  = self._render_hists(all_hist_data, initial_key)

        # ── Split slider callback ─────────────────────────────────────────────
        # split_values is a Python list serialised into JS as an array.
        # The slider value is an index (0...N-1); we look up the actual split value.
        # After filtering the main source, the same filtered dict (nd) is reused
        # to update every overlay source -- no extra per-overlay loop needed.
        split_cb = CustomJS(
            args=dict(
                full_source=self.full_source,
                display_source=self.source,
                split_values=self.split_values,
                frame_col=frame_col,
                mapper=self.color_mapper,
                plot_title=self.fig.title,
                scopeid=self.scopeid,
                overlay_sources=self.overlay_sources,
                overlay_xcols=overlay_xcols,
                overlay_ycols=overlay_ycols,
                hist_sources=hist_sources,
                all_hist_data=all_hist_data,
            ),
            code="""
            const idx       = Math.round(cb_obj.value);
            const split_val = split_values[idx];

            // Filter full_source to the selected split.
            const full = full_source.data;
            const nd   = {};
            for (const col of Object.keys(full)) { nd[col] = []; }
            for (let i = 0; i < full['split'].length; i++) {
                if (Number(full['split'][i]) === Number(split_val)) {
                    for (const col of Object.keys(full)) { nd[col].push(full[col][i]); }
                }
            }
            display_source.data = nd;

            // Update color-mapper range to this split's frame extent.
            const fd = nd[frame_col];
            if (fd && fd.length > 0) {
                mapper.low  = Math.min(...fd);
                mapper.high = Math.max(...fd);
            }

            // Update each overlay source from the same filtered dict.
            for (let i = 0; i < overlay_sources.length; i++) {
                const xc = overlay_xcols[i];
                const yc = overlay_ycols[i];
                overlay_sources[i].data = {
                    [xc]: nd[xc] !== undefined ? nd[xc] : [],
                    [yc]: nd[yc] !== undefined ? nd[yc] : [],
                };
            }

            // Update pre-computed histogram sources for this split.
            const hist_key = String(Math.round(split_val));
            if (all_hist_data[hist_key] !== undefined) {
                for (let i = 0; i < hist_sources.length; i++) {
                    if (all_hist_data[hist_key][i] !== undefined) {
                        hist_sources[i].data = all_hist_data[hist_key][i];
                    }
                }
            }

            plot_title.text = 'trappytv \u2014 ' + scopeid + ' \u2014 split ' + split_val;
            """,
        )

        self._add_fov()
        self._finalize(
            title=f"trappytv :: {self.scopeid} :: split {initial_split_val}",
            hover_columns=hover_columns,
            split_callback=split_cb,
            wire_checkboxes=True,
        )

    def view_all(
        self,
        sample: int = 10,
        xycols: Optional[Sequence[str]] = None,
        line_alpha: float = 0.4,
        line_color: str = "gray",
        hover_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Render the full trajectory with a sampled interactive scatter.

        A static grey line spans all data; the scatter (and side panels) show
        every ``sample``-th row and respond to interactive selection.
        The split slider and checkboxes are disabled in this mode.

        Parameters
        ----------
        sample : int
            Step for sub-sampling: every Nth row of self.df.
        xycols : [x_col, y_col], optional
            Overrides default_xycols.
        """
        xycols    = list(xycols or self.default_xycols)
        frame_col = "gframe_"

        self._clear_renderers()
        self.source = ColumnDataSource(self.df[::sample])

        color = transform(frame_col, self._make_color_mapper(frame_col))
        self._render_glyphs(
            xycols,
            line_source=(self.df[xycols[0]], self.df[xycols[1]]),
            scatter_source=self.source,
            scatter_color=color,
            line_alpha=line_alpha,
            line_color=line_color,
        )
        self.frame_col = frame_col   # stored for _wire_side_selects in _finalize
        self._render_sides(self.source, frame_col=frame_col, background_df=self.df)

        # Histograms — full-df data ("all" key); split slider is disabled in this
        # mode so there is no JS callback to wire. The histograms are static here.
        all_hist_data = self._precompute_split_hists()
        self._render_hists(all_hist_data, initial_key="all")

        # FOV — idempotent; safe to call on every view_all invocation.
        self._add_fov()

        self._finalize(
            title=f"trappytv :: {self.scopeid} :: full view (every {sample}th point)",
            hover_columns=hover_columns,
            disable_split_slider=True,
            disable_checkboxes=True,
        )

    def view_ensemble(
        self,
        xycols: Sequence[str] = ("x", "y"),
        smooth_window: int = 25,
        exclude_open: bool = False,
        hover_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Render all particles in an ensemble.

        Each (particle, scopeid) pair is assigned a unique colour per split.
        Speed is computed per particle, rolling-averaged, and shown on the first
        side panel. The split slider drives live JS updates.
        Checkboxes are disabled in this mode.

        Parameters
        ----------
        xycols : [x_col, y_col]
            Trajectory columns; typically ("x", "y").
        smooth_window : int
            Rolling average window in frames for speed smoothing.
        exclude_open : bool
            If True, rows where trap_open == True are excluded.
        hover_columns : sequence of str, optional
            Extra hover columns; scopeid/colony/particle are added automatically.
        """
        self._clear_renderers()

        prep       = self._prepare_ensemble(xycols, smooth_window, exclude_open)
        mode_df    = prep["mode_df"]
        frame_col  = prep["frame_col"]
        all_speed  = prep["all_speed"]
        initial    = prep["initial"]
        init_speed = prep["init_speed"]

        # ── Sources ───────────────────────────────────────────────────────────
        self.full_source    = ColumnDataSource(mode_df)
        self.display_source = ColumnDataSource(initial)
        self.source         = self.display_source

        self.speed_source = ColumnDataSource(data=dict(
            xs=init_speed["xs"], ys=init_speed["ys"], colors=init_speed["colors"],
        ))

        # ── Main glyphs ───────────────────────────────────────────────────────
        _empty = ColumnDataSource({xycols[0]: [], xycols[1]: []})
        self.line = self.fig.line(
            x=xycols[0], y=xycols[1], source=_empty, alpha=0, line_width=0,
        )
        self.scatter = self.fig.scatter(
            x=xycols[0], y=xycols[1], source=self.display_source,
            size=0.10, alpha=0.6, color="color", nonselection_alpha=0.0,
        )

        # ── Speed panel (first side figure) ──────────────────────────────────
        if self.side_figs:
            speed_fig = self.side_figs[0]
            speed_fig.renderers = []
            speed_fig.multi_line(
                xs="xs", ys="ys", source=self.speed_source,
                line_color="colors", line_width=1.5, alpha=0.8,
            )
            speed_fig.xaxis.axis_label = frame_col
            speed_fig.yaxis.axis_label = f"speed (rolling {smooth_window} fr)"
            speed_fig.title.text       = f"Speed (window={smooth_window})"

        _blank = ColumnDataSource({"x": [], "y": []})
        for fig in self.side_figs[1:]:
            fig.renderers = []
            fig.scatter(x="x", y="y", source=_blank, alpha=0, size=0)

        # ── Histograms ────────────────────────────────────────────────────────
        all_hist_data = self._precompute_split_hists()
        initial_key   = str(int(prep["initial_split"]))
        hist_sources  = self._render_hists(all_hist_data, initial_key)

        # ── FOV ───────────────────────────────────────────────────────────────
        self._add_fov()

        # ── Split slider callback ─────────────────────────────────────────────
        split_cb = CustomJS(
            args=dict(
                display_source=self.display_source,
                full_source=self.full_source,
                speed_source=self.speed_source,
                all_speed=all_speed,
                plot_title=self.fig.title,
                hist_sources=hist_sources,
                all_hist_data=all_hist_data,
            ),
            code="""
            const split_val = Number(cb_obj.value);
            const split_key = String(Math.round(split_val));
            const full = full_source.data;

            const nd = {};
            for (const col of Object.keys(full)) { nd[col] = []; }
            for (let i = 0; i < full['split'].length; i++) {
                if (Number(full['split'][i]) === split_val) {
                    for (const col of Object.keys(full)) { nd[col].push(full[col][i]); }
                }
            }
            display_source.data = nd;

            const sd = all_speed[split_key];
            if (sd !== undefined) {
                speed_source.data = { xs: sd['xs'], ys: sd['ys'], colors: sd['colors'] };
            }

            // Update pre-computed histogram sources for this split.
            if (all_hist_data[split_key] !== undefined) {
                for (let i = 0; i < hist_sources.length; i++) {
                    if (all_hist_data[split_key][i] !== undefined) {
                        hist_sources[i].data = all_hist_data[split_key][i];
                    }
                }
            }

            plot_title.text = 'trappytv \u2014 ensemble \u2014 split ' + split_key;
            """,
        )

        extra    = [c for c in ("scopeid", "colony", "particle") if c in mode_df.columns]
        base     = list(hover_columns or [])
        combined = (base + [c for c in extra if c not in base]) or None

        self.frame_col = frame_col   # stored for _wire_side_selects in _finalize

        self._finalize(
            title=f"trappytv :: {self.scopeid} :: ensemble :: split {self.split_no}",
            hover_columns=combined,
            split_callback=split_cb,
            disable_checkboxes=True,
        )

    # ── Ensemble data preparation ─────────────────────────────────────────────

    def _prepare_ensemble(
        self,
        xycols: Sequence[str],
        smooth_window: int,
        exclude_open: bool,
    ) -> dict:
        """
        Build a colour-annotated dataframe and per-split speed data for ensemble mode.
        Always operates on a local copy of self.df -- the original is never mutated.
        """
        mode_df = self.df.copy()

        if exclude_open and "trap_open" in mode_df.columns:
            mode_df = mode_df[mode_df["trap_open"] != True]

        max_pairs = int(
            mode_df[["split", "particle", "scopeid"]]
            .drop_duplicates()
            .groupby("split", observed=True)
            .size()
            .max()
        )
        if max_pairs <= 2:
            palette = ["#1f77b4", "#ff7f0e"]
        elif max_pairs <= 20:
            palette = list(Category20[max(3, max_pairs)])
        else:
            step    = max(1, 256 // max_pairs)
            palette = [Turbo256[i * step] for i in range(max_pairs)]

        combos = (
            mode_df[["split", "particle", "scopeid"]]
            .drop_duplicates()
            .copy()
        )
        combos["_rank"] = combos.groupby("split", observed=True).cumcount()
        combos["color"] = combos["_rank"].map(lambda r: palette[r % len(palette)])
        combos    = combos.drop(columns="_rank")
        mode_df   = mode_df.merge(combos, on=["split", "particle", "scopeid"], how="left")
        frame_col = "frame" if "frame" in mode_df.columns else "gframe_"

        all_speed: dict = {}
        for split_val, grp in mode_df.groupby("split", observed=True):
            xs, ys, colors = [], [], []
            for _, pgrp in grp.groupby(["particle", "scopeid"], sort=False, observed=True):
                pgrp = pgrp.sort_values(frame_col)
                fx   = pgrp[xycols[0]].to_numpy(dtype=float)
                fy   = pgrp[xycols[1]].to_numpy(dtype=float)
                ft   = pgrp[frame_col].to_numpy(dtype=float)

                dt = np.diff(ft)
                dt[dt == 0] = np.nan

                raw      = np.sqrt(np.diff(fx) ** 2 + np.diff(fy) ** 2) / dt
                raw      = np.concatenate([[np.nan], raw])
                smoothed = (
                    pd.Series(raw)
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
        initial       = mode_df[mode_df["split"] == initial_split]
        init_speed    = all_speed[str(initial_split)]

        return {
            "mode_df":       mode_df,
            "frame_col":     frame_col,
            "all_speed":     all_speed,
            "initial_split": initial_split,
            "initial":       initial,
            "init_speed":    init_speed,
        }

    def _precompute_split_hists(self) -> dict:
        """
        Pre-compute VBar data for every split (and the full df under key "all").

        The output is a plain Python dict that Bokeh's CustomJS serialiser can
        embed directly into a JS callback, exactly as ``all_speed`` is handled
        in view_ensemble.

        Returns
        -------
        all_hist_data : dict
            ``{split_key: [data_dict_per_hist_fig, ...]}``.
            *split_key* is ``str(int(split_val))`` or ``"all"``.
            Each inner list has exactly ``self._n_hists`` entries; unused figures
            receive empty arrays so the JS update loop stays uniform.

        Data dict shapes
        ----------------
        Figures 0 & 1 (subpixel bias):   ``{bins: [...], top: [...]}``
        Figures 2 +   (filter residuals): ``{xbins:[…], xtop:[…], ybins:[…], ytop:[…]}``
        """
        from .extraplots import subpix_hist, residual_hist

        # Detect unrefined / raw position columns (new spec name first).
        x_base = next((c for c in ("x_ur", "x_unrefined") if c in self.columns), None)
        y_base = next((c for c in ("y_ur", "y_unrefined") if c in self.columns), None)

        # Valid filtered-column pairs available in self.df.
        valid_fc = [
            (label, cols[0], cols[1])
            for label, cols in self.filtered_columns.items()
            if cols[0] in self.columns and cols[1] in self.columns
        ]

        # Build per-split sub-frames plus the "all" frame.
        if "split" in self.df.columns:
            groups: dict = {str(int(s)): self.df[self.df["split"] == s]
                            for s in self.split_values}
        else:
            groups = {}
        groups["all"] = self.df

        # ── Helper functions ──────────────────────────────────────────────────
        def _subpix(col: str, sub) -> dict:
            try:
                arr = sub[col].dropna().to_numpy(dtype=float)
                top, bins = subpix_hist(arr)
                return dict(bins=list(bins), top=list(top))
            except Exception:
                return dict(bins=[], top=[])

        def _residual(xf_col: str, yf_col: str, sub) -> dict:
            try:
                needed = [x_base, y_base, xf_col, yf_col]
                clean  = sub[needed].dropna()
                rx = clean[x_base].to_numpy() - clean[xf_col].to_numpy()
                ry = clean[y_base].to_numpy() - clean[yf_col].to_numpy()
                hx, bx = residual_hist(rx)
                hy, by = residual_hist(ry)
                return dict(xbins=list(bx), xtop=list(hx), ybins=list(by), ytop=list(hy))
            except Exception:
                return dict(xbins=[], xtop=[], ybins=[], ytop=[])

        _empty_sub = dict(bins=[], top=[])
        _empty_res = dict(xbins=[], xtop=[], ybins=[], ytop=[])

        # ── Compute ───────────────────────────────────────────────────────────
        all_hist_data: dict = {}
        for key, sub in groups.items():
            fig_data: list = []

            # fig 0 — subpixel bias x
            fig_data.append(_subpix(x_base, sub) if x_base else _empty_sub.copy())

            # fig 1 — subpixel bias y
            fig_data.append(_subpix(y_base, sub) if y_base else _empty_sub.copy())

            # fig 2 … — residuals per valid filtered column
            for i, (_, xf_col, yf_col) in enumerate(valid_fc):
                if 2 + i >= self._n_hists:
                    break
                if x_base and y_base:
                    fig_data.append(_residual(xf_col, yf_col, sub))
                else:
                    fig_data.append(_empty_res.copy())

            # Pad to self._n_hists so the JS update loop can index uniformly.
            while len(fig_data) < self._n_hists:
                fig_data.append(_empty_sub.copy())

            all_hist_data[key] = fig_data

        return all_hist_data

    def _render_hists(self, all_hist_data: dict, initial_key: str) -> list:
        """
        Clear hist_figs and populate them from pre-computed data, using
        .vbar(..., source=) so that each figure's ColumnDataSource can be
        updated from a JS split-slider callback without a Python round-trip.

        Parameters
        ----------
        all_hist_data : dict
            Output of _precompute_split_hists().
        initial_key : str
            The split key (e.g. "0" or "all") whose data to display first.

        Returns
        -------
        hist_sources : list[ColumnDataSource]
            One source per hist_fig.  Store these on self and pass them (along
            with all_hist_data) to the split-slider CustomJS callback.
        """
        from bokeh.palettes import Dark2_5 as palette

        x_base = next((c for c in ("x_ur", "x_unrefined") if c in self.columns), None)
        y_base = next((c for c in ("y_ur", "y_unrefined") if c in self.columns), None)

        valid_fc = [
            (label, cols[0], cols[1])
            for label, cols in self.filtered_columns.items()
            if cols[0] in self.columns and cols[1] in self.columns
        ]

        # Use the requested initial key, fall back to "all".
        initial_data = all_hist_data.get(initial_key, all_hist_data.get("all", []))

        hist_sources: list = []

        for i, fig in enumerate(self.hist_figs):
            fig.renderers = []          # clear any previous render

            init   = initial_data[i] if i < len(initial_data) else {}
            source = ColumnDataSource(init)
            hist_sources.append(source)

            if i == 0 and x_base:
                fig.vbar(x="bins", top="top", bottom=0, width=0.08,
                         fill_color="#b3de69", source=source)
                fig.title.text       = f"subpix_bias({x_base})"
                fig.x_range.bounds   = (0, 1)

            elif i == 1 and y_base:
                fig.vbar(x="bins", top="top", bottom=0, width=0.08,
                         fill_color="#b3de69", source=source)
                fig.title.text       = f"subpix_bias({y_base})"
                fig.x_range.bounds   = (0, 1)

            elif 2 <= i < 2 + len(valid_fc):
                label = valid_fc[i - 2][0]
                fig.vbar(x="xbins", top="xtop", bottom=0, width=0.01,
                         fill_color=palette[0], fill_alpha=0.3, line_color=None,
                         source=source)
                fig.vbar(x="ybins", top="ytop", bottom=0, width=0.01,
                         fill_color=palette[1], fill_alpha=0.3, line_color=None,
                         source=source)
                fig.title.text = f"residual: {label}"

            # else: blank placeholder — leave fig.renderers empty

        self.hist_sources = hist_sources
        return hist_sources

    def _add_fov(self):
        """
        Add the FOV image to self.fig exactly once (idempotent).

        On first call the image glyph is created with ``visible=False``; the
        "FOV image" checkbox (wired in _finalize via _wire_trap_fov_checkboxes)
        toggles it.  Because the renderer is static and not connected to any
        ColumnDataSource, the split slider — which only mutates
        display_source.data — can never trigger a re-render.

        Subsequent calls return the cached renderer without touching self.fig.

        Returns
        -------
        renderer : GlyphRenderer or None
        """
        # Already in the figure — return the cached handle.
        existing = getattr(self, "fov_renderer", None)
        if existing is not None and existing in self.fig.renderers:
            return existing

        if self.fov is None:
            self.fov_renderer = None
            return None

        img = self.fov.astype(float)
        h, w = img.shape[:2]

        color_mapper = LinearColorMapper(
            low=float(img.min()), high=float(img.max()), palette="Greys256"
        )
        renderer = self.fig.image(
            image=[img], x=0, y=0, dw=w, dh=h,
            color_mapper=color_mapper,
            level="image",
            visible=False,   # hidden until the user enables the checkbox
        )
        self.fov_renderer = renderer
        return renderer


