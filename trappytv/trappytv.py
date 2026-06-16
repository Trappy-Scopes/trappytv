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
from bokeh.models import Circle, ColumnDataSource, CustomJS, HoverTool
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
        compute_hists: bool = True,
        trap_radius_col: str = "rout",
        scale_bar_unit: Optional[str] = None,
    ) -> None:
        super().__init__(
            cell=cell,
            width=width,
            height=height,
            default_xycols=default_xycols,
            filtered_columns=filtered_columns,
            side_cols=side_cols,
            compute_hists=compute_hists,
            trap_radius_col=trap_radius_col,
            scale_bar_unit=scale_bar_unit,
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

        # Histogram state -- populated by _render_hists on every view call.
        # _hist_bin_info_js holds global bin edges (computed once from full df)
        # used by the JS selection and split-slider callbacks.
        self._hist_bin_info_js: dict = {}

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

        for i, (fig, (y_col_default, label)) in enumerate(zip(self.side_figs, self.side_cols)):
            fig.renderers = []

            # Use the select's current value — it already defaults to a valid column
            # at build time if the side_cols default doesn't exist in the data.
            # Fall back to the side_cols default, then skip if neither exists.
            sel_val = (
                self.side_selects[i].value
                if i < len(self.side_selects) and self.side_selects[i].value in self.df.columns
                else y_col_default
            )
            y_col = sel_val if sel_val in self.df.columns else y_col_default

            fig.xaxis.axis_label = frame_col
            fig.yaxis.axis_label = y_col

            if y_col not in self.df.columns:
                continue

            if (background_df is not None
                    and frame_col in background_df.columns
                    and y_col in background_df.columns):
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

        # FOV controls — wired here so all view modes benefit.
        # _add_fov() is idempotent; self.fov_renderer / self._fov_renderers_dict
        # are populated before _finalize is called.
        self._wire_trap_fov_checkboxes(
            trap_renderer=self.trap_renderer,
            fov_renderer=self.fov_renderer,
        )
        self._wire_fov_select()

        # Histogram selection callback — wired after every view call so it
        # always points at the current self.source.
        self._wire_hist_selection()

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

        # Resolve split_no: accept either an actual split VALUE (preferred)
        # or a positional index into split_values.  Lets callers pass
        # split_no=5 to mean "the split whose value is 5" — works correctly
        # even when splits are discontinuous or don't start from zero.
        if split_no in self.split_values:
            initial_split_val = split_no
            slider_idx        = self.split_values.index(split_no)
        elif isinstance(split_no, int) and 0 <= split_no < len(self.split_values):
            initial_split_val = self.split_values[split_no]
            slider_idx        = split_no
        else:
            initial_split_val = self.split_values[0]
            slider_idx        = 0

        self.split_no           = slider_idx
        self.split_slider.value = slider_idx
        self.split_slider.title = f"Split ({initial_split_val})"

        initial_df = (
            self.df[self.df["split"] == initial_split_val]
            if "split" in self.df.columns
            else self.df
        )

        # full_source: all splits -- never mutated; used by the JS split callback.
        # display_source (self.source): the currently visible split; rewritten by JS.
        self.full_source = ColumnDataSource(self.df)
        self.source      = ColumnDataSource(initial_df)

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
        # Compute global bin edges once from the full df; pass to JS so the
        # split callback and selection callback can bin in real time without
        # serialising per-split counts.
        bin_info     = self._compute_hist_bin_info()
        hist_sources = self._render_hists(bin_info, initial_df)

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
                hist_specs=self._hist_specs,
                bin_info=bin_info,
                trap_source=self.trap_source,
                xyr_source=self.xyr_source,
                trap_radius_col=self._trap_radius_col,
                split_slider=self.split_slider,
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


            // Update trap boundary circle to this split.
            if (trap_source !== null && xyr_source !== null && trap_radius_col !== null) {
                const xyr = xyr_source.data;
                const td  = { xc: [], yc: [], radius: [] };
                if ('split' in xyr) {
                    for (let i = 0; i < xyr['split'].length; i++) {
                        if (Number(xyr['split'][i]) === Number(split_val)) {
                            td.xc.push(xyr['xc'][i]);
                            td.yc.push(xyr['yc'][i]);
                            td.radius.push(xyr[trap_radius_col][i]);
                        }
                    }
                }
                if (td.xc.length > 0) trap_source.data = td;
            }
            // Recompute histograms in JS from the new split data.
            // Clear any lingering point selection so indices stay valid.
            display_source.selected.indices = [];

            function bin_vals(vals, bi, transform) {
                const top = new Array(bi.n).fill(0); let count = 0;
                for (let j = 0; j < vals.length; j++) {
                    let v = Number(vals[j]); if (!isFinite(v)) continue;
                    if (transform === 'frac') v = v - Math.floor(v);
                    const b = Math.floor((v - bi.min) / bi.width);
                    if (b >= 0 && b < bi.n) { top[b]++; count++; }
                }
                if (count > 0 && bi.width > 0)
                    for (let b = 0; b < bi.n; b++) top[b] /= (count * bi.width);
                return top;
            }

            for (let i = 0; i < hist_specs.length; i++) {
                const spec = hist_specs[i];
                const bi   = bin_info[String(i)];
                const hsrc = hist_sources[i];
                if (!bi) continue;

                if (spec.type === 'subpix') {
                    if (!(spec.col in nd)) continue;
                    hsrc.data = { left: bi.left, right: bi.right,
                                  top: bin_vals(nd[spec.col], bi, 'frac') };

                } else if (spec.type === 'residual') {
                    if (!(spec.xraw in nd) || !(spec.xfilt in nd)) continue;
                    const xr = nd[spec.xraw],  xf = nd[spec.xfilt];
                    const yr = nd[spec.yraw],  yf = nd[spec.yfilt];
                    const rx = new Array(xr.length), ry = new Array(yr.length);
                    for (let j = 0; j < xr.length; j++) rx[j] = Number(xr[j]) - Number(xf[j]);
                    for (let j = 0; j < yr.length; j++) ry[j] = Number(yr[j]) - Number(yf[j]);
                    hsrc.data = { left: bi.left, right: bi.right,
                                  xtop: bin_vals(rx, bi, 'id'),
                                  ytop: bin_vals(ry, bi, 'id') };
                }
            }
            split_slider.title = 'Split (' + split_val + ')';
            plot_title.text = 'trappytv \u2014 ' + scopeid + ' \u2014 split ' + split_val;
            """,
        )

        self._add_trap(initial_split_val)
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
        The split slider is disabled in this mode. Filtered-column overlay
        checkboxes are active (overlays are drawn from the sampled df, static).

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
        sampled_df    = self.df[::sample]
        self.source   = ColumnDataSource(sampled_df)

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

        # Filtered-column overlays — drawn from the sampled df; static in this
        # mode (no split slider), checkboxes toggle visibility as normal.
        self._render_filtered_overlays(sampled_df)

        # Histograms — computed from the sampled df for first render;
        # the selection callback updates them on point selection.
        bin_info = self._compute_hist_bin_info()
        self._render_hists(bin_info, sampled_df)

        # Trap boundary — first split only (no slider in view_all).
        self._add_trap(self.split_values[0] if self.split_values else None)
        # FOV — idempotent; safe to call on every view_all invocation.
        self._add_fov()

        self._finalize(
            title=f"trappytv :: {self.scopeid} :: full view (every {sample}th point)",
            hover_columns=hover_columns,
            disable_split_slider=True,
            wire_checkboxes=True,
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

        # ── Filtered-column overlays ──────────────────────────────────────────
        valid_overlays = self._render_filtered_overlays(initial)
        overlay_xcols  = [x for _, x, _ in valid_overlays]
        overlay_ycols  = [y for _, _, y in valid_overlays]

        # ── Histograms ────────────────────────────────────────────────────────
        bin_info     = self._compute_hist_bin_info()
        hist_sources = self._render_hists(bin_info, initial)

        # ── FOV / Trap ────────────────────────────────────────────────────────
        # _add_trap handles dict xyr gracefully (returns without rendering).
        self._add_trap(prep["initial_split"])
        self._add_fov()

        # ── Split slider callback ─────────────────────────────────────────────
        # BUG FIX: pass split_values so JS can convert slider index -> real value.
        split_cb = CustomJS(
            args=dict(
                display_source=self.display_source,
                full_source=self.full_source,
                speed_source=self.speed_source,
                all_speed=all_speed,
                split_values=self.split_values,
                plot_title=self.fig.title,
                overlay_sources=self.overlay_sources,
                overlay_xcols=overlay_xcols,
                overlay_ycols=overlay_ycols,
                hist_sources=hist_sources,
                hist_specs=self._hist_specs,
                bin_info=bin_info,
                trap_source=self.trap_source,
                xyr_source=self.xyr_source,
                trap_radius_col=self._trap_radius_col,
                split_slider=self.split_slider,
            ),
            code="""
            const idx       = Math.round(cb_obj.value);
            const split_val = split_values[idx];
            const split_key = String(Math.round(split_val));
            const full      = full_source.data;

            // Filter full_source to the selected split.
            const nd = {};
            for (const col of Object.keys(full)) { nd[col] = []; }
            for (let i = 0; i < full['split'].length; i++) {
                if (Number(full['split'][i]) === Number(split_val)) {
                    for (const col of Object.keys(full)) { nd[col].push(full[col][i]); }
                }
            }
            display_source.data = nd;

            const sd = all_speed[split_key];
            if (sd !== undefined) {
                speed_source.data = { xs: sd['xs'], ys: sd['ys'], colors: sd['colors'] };
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


            // Update trap boundary circle to this split.
            if (trap_source !== null && xyr_source !== null && trap_radius_col !== null) {
                const xyr = xyr_source.data;
                const td  = { xc: [], yc: [], radius: [] };
                if ('split' in xyr) {
                    for (let i = 0; i < xyr['split'].length; i++) {
                        if (Number(xyr['split'][i]) === Number(split_val)) {
                            td.xc.push(xyr['xc'][i]);
                            td.yc.push(xyr['yc'][i]);
                            td.radius.push(xyr[trap_radius_col][i]);
                        }
                    }
                }
                if (td.xc.length > 0) trap_source.data = td;
            }
            // Recompute histograms in JS from the new split data.
            // Clear any lingering point selection so indices stay valid.
            display_source.selected.indices = [];

            function bin_vals(vals, bi, transform) {
                const top = new Array(bi.n).fill(0); let count = 0;
                for (let j = 0; j < vals.length; j++) {
                    let v = Number(vals[j]); if (!isFinite(v)) continue;
                    if (transform === 'frac') v = v - Math.floor(v);
                    const b = Math.floor((v - bi.min) / bi.width);
                    if (b >= 0 && b < bi.n) { top[b]++; count++; }
                }
                if (count > 0 && bi.width > 0)
                    for (let b = 0; b < bi.n; b++) top[b] /= (count * bi.width);
                return top;
            }

            for (let i = 0; i < hist_specs.length; i++) {
                const spec = hist_specs[i];
                const bi   = bin_info[String(i)];
                const hsrc = hist_sources[i];
                if (!bi) continue;

                if (spec.type === 'subpix') {
                    if (!(spec.col in nd)) continue;
                    hsrc.data = { left: bi.left, right: bi.right,
                                  top: bin_vals(nd[spec.col], bi, 'frac') };

                } else if (spec.type === 'residual') {
                    if (!(spec.xraw in nd) || !(spec.xfilt in nd)) continue;
                    const xr = nd[spec.xraw],  xf = nd[spec.xfilt];
                    const yr = nd[spec.yraw],  yf = nd[spec.yfilt];
                    const rx = new Array(xr.length), ry = new Array(yr.length);
                    for (let j = 0; j < xr.length; j++) rx[j] = Number(xr[j]) - Number(xf[j]);
                    for (let j = 0; j < yr.length; j++) ry[j] = Number(yr[j]) - Number(yf[j]);
                    hsrc.data = { left: bi.left, right: bi.right,
                                  xtop: bin_vals(rx, bi, 'id'),
                                  ytop: bin_vals(ry, bi, 'id') };
                }
            }

            split_slider.title = 'Split (' + split_val + ')';
            plot_title.text = 'trappytv \u2014 ensemble \u2014 split ' + split_key;
            """,
        )

        extra    = [c for c in ("scopeid", "colony", "particle") if c in mode_df.columns]
        base     = list(hover_columns or [])
        combined = (base + [c for c in extra if c not in base]) or None

        self.frame_col              = frame_col  # stored for _wire_side_selects in _finalize
        self.split_no               = 0
        self.split_slider.value     = 0
        self.split_slider.title     = f"Split ({prep['initial_split']})"

        self._finalize(
            title=f"trappytv :: {self.scopeid} :: ensemble :: split {prep['initial_split']}",
            hover_columns=combined,
            split_callback=split_cb,
            wire_checkboxes=True,
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

    def _compute_hist_bin_info(self) -> dict:
        """
        Compute bin edges for each histogram spec (one pass, fast).

        Sub-pixel bias specs use fixed [0, 1) edges (no df scan needed).
        Residual specs derive edges from the 1st–99th percentile of the
        combined x+y residuals over the full dataframe.

        Returns ``{str(i): {left, right, min, width, n}}`` keyed by spec index
        so the JS callbacks can look up the right edges for each figure.
        """
        if not self._compute_hists or not self.hist_figs:
            return {}

        n_subpix   = 10
        n_residual = 40
        bin_info: dict = {}

        for i, spec in enumerate(self._hist_specs):
            key = str(i)
            if spec["type"] == "subpix":
                w = 1.0 / n_subpix
                edges = np.linspace(0.0, 1.0, n_subpix + 1)
                bin_info[key] = {
                    "left":  edges[:-1].tolist(),
                    "right": edges[1:].tolist(),
                    "min":   0.0,
                    "width": w,
                    "n":     n_subpix,
                }
            elif spec["type"] == "residual":
                try:
                    needed = [spec["xraw"], spec["yraw"], spec["xfilt"], spec["yfilt"]]
                    clean  = self.df[needed].dropna()
                    rx = clean[spec["xraw"]].to_numpy() - clean[spec["xfilt"]].to_numpy()
                    ry = clean[spec["yraw"]].to_numpy() - clean[spec["yfilt"]].to_numpy()
                    combined = np.concatenate([rx, ry])
                    finite   = combined[np.isfinite(combined)]
                    if len(finite) < 2:
                        continue
                    q1, q99 = np.percentile(finite, [1, 99])
                    # Robust range: if the 1–99 percentile range is degenerate
                    # (e.g. filter barely moves from raw), widen to full extent
                    # with a small padding so all data still gets a bin.
                    if abs(q99 - q1) < 1e-10:
                        lo, hi = float(finite.min()), float(finite.max())
                        if lo == hi:          # truly constant residuals
                            lo, hi = lo - 0.5, hi + 0.5
                    else:
                        lo, hi = float(q1), float(q99)
                    _, edges = np.histogram(finite, bins=n_residual, range=(lo, hi))
                    w = float(edges[1] - edges[0])
                    bin_info[key] = {
                        "left":  edges[:-1].tolist(),
                        "right": edges[1:].tolist(),
                        "min":   float(edges[0]),
                        "width": w,
                        "n":     n_residual,
                    }
                except Exception:
                    pass

        return bin_info

    def _render_hists(self, bin_info: dict, initial_df) -> list:
        """
        Render the initial histogram for ``initial_df`` and store sources.

        Sub-pixel bias figures get one ``quad`` glyph (``{left, right, top}``).
        Residual figures get two overlapping ``quad`` glyphs sharing the same
        bins (``{left, right, xtop, ytop}``): x-residual in blue, y in orange.

        ``bin_info`` is the output of ``_compute_hist_bin_info()`` (keyed by
        spec index string).  Stored as ``self._hist_bin_info_js`` for the
        selection callback wired in ``_finalize``.
        """
        if not self._compute_hists or not self.hist_figs:
            self.hist_sources      = []
            self._hist_bin_info_js = {}
            return []

        self._hist_bin_info_js = bin_info

        _subpix_color  = "#b3de69"
        _resid_x_color = "#377eb8"   # blue  — x residual
        _resid_y_color = "#e41a1c"   # red   — y residual
        _empty_sub = {"left": [], "right": [], "top": []}
        _empty_res = {"left": [], "right": [], "xtop": [], "ytop": []}

        def _density(counts: np.ndarray, bin_width: float) -> list:
            """Normalise raw bin counts to probability density, returning zeros
            (not NaN) when the array is empty or all-zero."""
            total = counts.sum()
            if total > 0 and bin_width > 0:
                return (counts / (total * bin_width)).tolist()
            return [0.0] * len(counts)

        hist_sources: list = []
        for i, (fig, spec) in enumerate(zip(self.hist_figs, self._hist_specs)):
            fig.renderers  = []
            fig.y_range.start = 0
            key = str(i)
            bi  = bin_info.get(key)

            if spec["type"] == "subpix":
                col = spec["col"]
                if bi and col in initial_df.columns:
                    arr  = initial_df[col].dropna().to_numpy(dtype=float)
                    frac = arr % 1
                    frac = frac[np.isfinite(frac)]
                    edges  = np.array(bi["left"] + [bi["right"][-1]])
                    counts, _ = np.histogram(frac, bins=edges)
                    init = {"left": bi["left"], "right": bi["right"],
                            "top": _density(counts, bi["width"])}
                else:
                    init = _empty_sub.copy()
                source = ColumnDataSource(init)
                fig.quad(left="left", right="right", top="top", bottom=0,
                         fill_color=_subpix_color, fill_alpha=0.8,
                         line_color="white", source=source)

            else:  # residual
                xraw, yraw   = spec["xraw"],  spec["yraw"]
                xfilt, yfilt = spec["xfilt"], spec["yfilt"]
                if bi and all(c in initial_df.columns for c in [xraw, yraw, xfilt, yfilt]):
                    clean  = initial_df[[xraw, yraw, xfilt, yfilt]].dropna()
                    rx     = clean[xraw].to_numpy()  - clean[xfilt].to_numpy()
                    ry     = clean[yraw].to_numpy()  - clean[yfilt].to_numpy()
                    edges  = np.array(bi["left"] + [bi["right"][-1]])
                    cx, _  = np.histogram(rx, bins=edges)
                    cy, _  = np.histogram(ry, bins=edges)
                    init   = {"left": bi["left"], "right": bi["right"],
                              "xtop": _density(cx, bi["width"]),
                              "ytop": _density(cy, bi["width"])}
                else:
                    init = _empty_res.copy()
                source = ColumnDataSource(init)
                # Both as solid semi-transparent fills so they're always visible.
                # Overlap region blends to purple, indicating x ≈ y distributions.
                fig.quad(left="left", right="right", top="xtop", bottom=0,
                         fill_color=_resid_x_color, fill_alpha=0.6,
                         line_color=None, source=source)
                fig.quad(left="left", right="right", top="ytop", bottom=0,
                         fill_color=_resid_y_color, fill_alpha=0.4,
                         line_color=None, source=source)

            fig.title.text = spec["title"]
            hist_sources.append(source)

        self.hist_sources = hist_sources
        return hist_sources

    def _wire_hist_selection(self) -> None:
        """
        Wire ``self.source.selected`` so that lasso/box selection updates the
        histogram row in real time.

        Sub-pixel bias: bins the fractional part of the raw column for selected
        points (or all points when selection is cleared).

        Residual: bins ``raw - filtered`` for the selected points; updates both
        x (blue) and y (red) bars in the residual figures.

        Clears any previous callback first to avoid stacking on re-renders.
        """
        if not self._compute_hists or not self.hist_sources or not self._hist_bin_info_js:
            return
        if not self._hist_specs:
            return

        self.source.selected.js_property_callbacks.pop("change:indices", None)

        sel_cb = CustomJS(
            args=dict(
                source=self.source,
                hist_sources=self.hist_sources,
                hist_specs=self._hist_specs,
                bin_info=self._hist_bin_info_js,
            ),
            code="""
            const indices = cb_obj.indices;
            const data    = source.data;
            const use_all = (indices.length === 0);

            function extract(col, indices, use_all, data) {
                if (!(col in data)) return [];
                const all = data[col];
                if (use_all) return all;
                const out = new Array(indices.length);
                for (let k = 0; k < indices.length; k++) out[k] = all[indices[k]];
                return out;
            }

            function bin_vals(vals, bi, transform) {
                // transform: "id" | "frac"
                const top = new Array(bi.n).fill(0); let count = 0;
                for (let j = 0; j < vals.length; j++) {
                    let v = Number(vals[j]); if (!isFinite(v)) continue;
                    if (transform === "frac") v = v - Math.floor(v);
                    const b = Math.floor((v - bi.min) / bi.width);
                    if (b >= 0 && b < bi.n) { top[b]++; count++; }
                }
                if (count > 0 && bi.width > 0)
                    for (let b = 0; b < bi.n; b++) top[b] /= (count * bi.width);
                return top;
            }

            for (let i = 0; i < hist_specs.length; i++) {
                const spec = hist_specs[i];
                const bi   = bin_info[String(i)];
                const src  = hist_sources[i];
                if (!bi) continue;

                if (spec.type === "subpix") {
                    const vals = extract(spec.col, indices, use_all, data);
                    const top  = bin_vals(vals, bi, "frac");
                    src.data   = { left: bi.left, right: bi.right, top: top };

                } else if (spec.type === "residual") {
                    if (!(spec.xraw in data) || !(spec.xfilt in data)) continue;
                    const xr = extract(spec.xraw,  indices, use_all, data);
                    const xf = extract(spec.xfilt, indices, use_all, data);
                    const yr = extract(spec.yraw,  indices, use_all, data);
                    const yf = extract(spec.yfilt, indices, use_all, data);
                    // compute residuals inline
                    const rx = new Array(xr.length), ry = new Array(yr.length);
                    for (let j = 0; j < xr.length; j++) rx[j] = Number(xr[j]) - Number(xf[j]);
                    for (let j = 0; j < yr.length; j++) ry[j] = Number(yr[j]) - Number(yf[j]);
                    const xtop = bin_vals(rx, bi, "id");
                    const ytop = bin_vals(ry, bi, "id");
                    src.data = { left: bi.left, right: bi.right, xtop: xtop, ytop: ytop };
                }
            }
            """,
        )
        self.source.selected.js_on_change("indices", sel_cb)


    def _add_trap(self, initial_split_val=None) -> None:
        """
        Render the trap boundary circle from ``self.xyr`` (single-cell mode only).

        The radius column is resolved in priority order:
          1. ``self.trap_radius_col`` (set by the ``trap_radius_col`` constructor arg,
             default ``"rout"``)
          2. Fallbacks: ``"rout"``, ``"reff"``, ``"r"`` — first one present wins

        ``trap_source`` uses the normalised key ``"radius"`` regardless of which
        column was resolved, so the ``Circle`` glyph and JS callbacks always reference
        ``"radius"`` and never need to know the actual column name.
        ``xyr_source`` holds the full DataFrame (tiny — one row per split) for the
        JS split-slider callbacks; ``self._trap_radius_col`` records the resolved name
        so the callbacks know which column to read from ``xyr_source``.

        Safe to call on every view render — always removes the old renderer first.

        Does nothing when:
          - ``self.xyr`` is None or a dict (ensemble mode)
          - ``xc`` / ``yc`` are missing
          - no usable radius column is found
        """
        # Remove any existing trap renderer so we start fresh each view call.
        if self.trap_renderer is not None and self.trap_renderer in self.fig.renderers:
            self.fig.renderers.remove(self.trap_renderer)
        self.trap_renderer     = None
        self.trap_source       = None
        self.xyr_source        = None
        self._trap_radius_col  = None  # resolved column name for JS callbacks

        xyr = self.xyr
        if xyr is None or not hasattr(xyr, "columns"):
            return  # None or dict (ensemble) — skip

        if not {"xc", "yc"}.issubset(set(xyr.columns)):
            return

        # ── Resolve radius column ─────────────────────────────────────────────
        _fallbacks = ["rout", "reff", "r"]
        candidates = [self.trap_radius_col] + [c for c in _fallbacks
                                                if c != self.trap_radius_col]
        rad_col = next((c for c in candidates if c in xyr.columns), None)
        if rad_col is None:
            return  # no usable radius column
        self._trap_radius_col = rad_col

        # ── Initial row for this split ────────────────────────────────────────
        if initial_split_val is not None and "split" in xyr.columns:
            row = xyr[xyr["split"] == initial_split_val]
            if row.empty:
                row = xyr.head(1)
        else:
            row = xyr.head(1)

        # Normalise radius key to "radius" so glyphs/JS never need the real name.
        self.trap_source = ColumnDataSource({
            "xc":     row["xc"].tolist(),
            "yc":     row["yc"].tolist(),
            "radius": row[rad_col].tolist(),
        })

        # ── Full xyr source for JS split-slider callbacks ─────────────────────
        keep = [c for c in ("xc", "yc", rad_col, "split") if c in xyr.columns]
        self.xyr_source = ColumnDataSource(xyr[keep])

        # ── Render ────────────────────────────────────────────────────────────
        # Use Circle glyph model directly so radius is in data coordinates.
        glyph = Circle(
            x="xc", y="yc",
            radius="radius",
            fill_color=None,
            line_color="orange",
            line_width=2,
            line_dash=[6, 3],
        )
        renderer = self.fig.add_glyph(self.trap_source, glyph)
        renderer.level   = "overlay"
        renderer.visible = True
        self.trap_renderer = renderer

    def _add_fov(self):
        """
        Add FOV image(s) to self.fig (idempotent).

        Single-cell mode (self.fov is np.ndarray):
            One hidden image renderer stored in self.fov_renderer.

        Ensemble mode (self.fov is dict[eid, np.ndarray]):
            One hidden image renderer per eid, stored in self._fov_renderers_dict.
            self.fov_select is populated with eid names so the user can pick which
            scope's FOV to show.  self.fov_renderer is set to None in this mode —
            fov_select + trap_fov_checkboxes control individual renderers.

        Subsequent calls are no-ops if renderers are already attached to self.fig.
        """
        if self.fov is None:
            self.fov_renderer = None
            return None

        # ── Ensemble mode ─────────────────────────────────────────────────────
        if isinstance(self.fov, dict):
            # Already built for this figure — nothing to do.
            if self._fov_renderers_dict and all(
                r in self.fig.renderers for r in self._fov_renderers_dict.values()
            ):
                return self._fov_renderers_dict

            # Remove stale renderers from a previous view call.
            for r in list(self._fov_renderers_dict.values()):
                if r in self.fig.renderers:
                    self.fig.renderers.remove(r)
            self._fov_renderers_dict = {}

            for eid, img_arr in self.fov.items():
                img = img_arr.astype(float)
                h, w = img.shape[:2]
                color_mapper = LinearColorMapper(
                    low=float(img.min()), high=float(img.max()), palette="Greys256"
                )
                renderer = self.fig.image(
                    image=[img], x=0, y=0, dw=w, dh=h,
                    color_mapper=color_mapper,
                    level="image",
                    visible=False,
                )
                self._fov_renderers_dict[str(eid)] = renderer

            self.fov_renderer = None  # controlled via fov_select, not directly

            if self._fov_renderers_dict:
                keys = list(self._fov_renderers_dict.keys())
                self.fov_select.options  = keys
                self.fov_select.value    = keys[0]
                self.fov_select.disabled = False
                self.fov_select.visible  = True

            return self._fov_renderers_dict

        # ── Single-cell mode ──────────────────────────────────────────────────
        existing = getattr(self, "fov_renderer", None)
        if existing is not None and existing in self.fig.renderers:
            return existing

        img = self.fov.astype(float)
        h, w = img.shape[:2]

        color_mapper = LinearColorMapper(
            low=float(img.min()), high=float(img.max()), palette="Greys256"
        )
        renderer = self.fig.image(
            image=[img], x=0, y=0, dw=w, dh=h,
            color_mapper=color_mapper,
            level="image",
            visible=False,
        )
        self.fov_renderer = renderer
        return renderer


