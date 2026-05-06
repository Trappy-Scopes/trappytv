"""
trappycore.py
─────────────
Infrastructure base for TrappyTV.

Responsibilities
----------------
- Data normalisation from a cell object
- Figure and side-figure creation
- Widget creation: sliders, column selects, checkboxes
- Layout assembly
- Shared renderer helpers: clear, color mapper factory, unified glyph creation

This class is not meant to be instantiated directly.
View logic (view_split, view_all, view_ensemble) lives in TrappyTV.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from bokeh.core.properties import value as bokeh_value
from bokeh.io import show as _bokeh_show
from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxButtonGroup,
    ColumnDataSource,
    CustomJS,
    ImageURL,
    LinearColorMapper,
    Select,
    Slider,
    Spacer,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure


# ── Module-level constants ────────────────────────────────────────────────────

_MAIN_TOOLS = "box_select,lasso_select,pan,box_zoom,wheel_zoom,reset,save,hover"
_SIDE_TOOLS = "box_select,lasso_select,pan,box_zoom,wheel_zoom,reset,save"
_LOGO_URL = (
    "https://github.com/Trappy-Scopes/Trappy-Scopes.github.io"
    "/blob/main/docs/assets/tsicon.png?raw=true"
)

# Default side-panel columns: (data_col, display_label)
_DEFAULT_SIDE_COLS: Tuple[Tuple[str, str], ...] = (
    ("speed",  "Speed"),
    ("signal", "Signal"),
    ("temp",   "Temp"),
)


# ── TrappyCore ────────────────────────────────────────────────────────────────

class TrappyCore:
    """
    Infrastructure base for TrappyTV.

    Parameters
    ----------
    cell : object
        Provides ``cell.dfs["tracks"]`` (DataFrame with a ``gframe`` column)
        and ``cell.scopeid`` (string label).
    width, height : int
        Pixel dimensions of the main figure.
    figs_width, figs_height : int
        Pixel dimensions of each side figure.
        ``figs_height`` defaults to ``height // 3``.
    filtered_columns : dict
        Mapping of label → [x_col, y_col] used to build overlay checkboxes.
        Pass ``{}`` (default) to suppress checkboxes.
    side_cols : sequence of (str, str)
        Pairs of (data_column, display_label) for the side panels.
        The number of pairs controls how many side figures are created.
    """

    def __init__(
        self,
        cell,
        width: int = 1000,
        height: int = 1000,
        figs_width: int = 140 * 5,
        figs_height: Optional[int] = None,
        filtered_columns: dict = {},
        side_cols: Sequence[Tuple[str, str]] = _DEFAULT_SIDE_COLS,
    ) -> None:
        # ── 1. Data ───────────────────────────────────────────────────────────
        self._load_cell(cell)

        # ── 2. Dimensions (stored for use in figure/layout builders) ──────────
        self._width      = width
        self._height     = height
        self._figs_width = figs_width
        self._figs_height = figs_height if figs_height is not None else height // 3

        self.filtered_columns = dict(filtered_columns)
        self.side_cols = list(side_cols)

        # ── 3. Build everything ───────────────────────────────────────────────
        self._build_figures()
        self._build_widgets()
        self._build_layout()

    # ── Data normalisation ────────────────────────────────────────────────────

    def _load_cell(self, cell) -> None:
        """
        Load tracking data from a cell object and normalise column names.
        Called once from ``__init__``; never mutates ``self.df`` afterward.
        """
        self.df = cell.dfs["tracks"].rename(columns={"gframe": "gframe_"})
        self.scopeid: str = cell.scopeid

        self.split_values: list = (
            sorted(self.df["split"].dropna().unique().tolist())
            if "split" in self.df.columns
            else [0]
        )
        self.split_no: int = 0

        # Canonical column list — kept in sync with self.df after _load_cell.
        self.columns: list[str] = list(self.df.columns)

    # ── Figure factory ────────────────────────────────────────────────────────

    def _build_figures(self) -> None:
        """Create ``self.fig`` (main) and ``self.side_figs`` (side panels)."""

        self.fig = figure(
            width=self._width,
            height=self._height,
            title="trappytv",
            tools=_MAIN_TOOLS,
            x_axis_label="x",
            y_axis_label="y",
            output_backend="webgl",
        )
        # Lock aspect ratio on any zoom tool that supports it.
        for tool in self.fig.tools:
            if hasattr(tool, "match_aspect"):
                tool.match_aspect = True

        self.side_figs: list = [
            figure(
                width=self._figs_width,
                height=self._figs_height,
                title=label,
                tools=_SIDE_TOOLS,
                output_backend="webgl",
            )
            for _, label in self.side_cols
        ]

        # Shared interactive source — view methods repopulate .data each call.
        initial_df = (
            self.df[self.df["split"] == self.split_values[0]]
            if "split" in self.df.columns
            else self.df
        )
        self.source = ColumnDataSource(initial_df)

        # Placeholder renderers so _clear_renderers() always has valid targets.
        # They are invisible; every view method replaces them before display.
        _empty = ColumnDataSource({"x": [], "y": []})
        self.scatter = self.fig.scatter("x", "y", source=_empty, alpha=0, size=0)
        self.line    = self.fig.line("x", "y", source=_empty, alpha=0, line_width=0)

        # Logo — positioned relative to figure height, not a hardcoded pixel value.
        self._add_logo()

        # Color mapper — initialised to None; created lazily by _make_color_mapper.
        self.color_mapper: Optional[LinearColorMapper] = None

        # Side-panel scatter renderers — populated by _render_sides each view call.
        self.other_scatters: list = []

    def _add_logo(self) -> None:
        logo = ImageURL(
            url=bokeh_value(_LOGO_URL),
            w_units="screen",
            h_units="screen",
            x=0,
            y=self._height + 20,
            w=64,
            h=25,
            anchor="bottom_left",
        )
        self.fig.add_glyph(logo)

    # ── Widget factory ────────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self._build_sliders()
        self._build_selects()
        self._build_raw_checkboxes()
        self._build_checkboxes()

    def _build_sliders(self) -> None:
        """Create alpha, size, and split sliders. Callbacks are wired later by
        _wire_style_sliders(), called from _finalize() after real renderers exist."""
        self.alpha_slider = Slider(
            start=0.0, end=1.0, value=0.4, step=0.05, title="Alpha", width=250
        )
        self.size_slider = Slider(
            start=0.001, end=20, value=0.1, step=0.1, title="Size", width=250
        )
        self.split_slider = Slider(
            start=0,
            end=max(len(self.split_values) - 1, 0),
            value=0,
            step=1,
            title="Split",
            width=250,
        )

    def _wire_style_sliders(self) -> None:
        """
        Attach alpha/size callbacks to the *current* self.scatter and self.other_scatters.
        Must be called from _finalize(), after every view method creates real renderers.
        Clears previously attached callbacks first to avoid stacking on re-renders.
        """
        for slider in (self.alpha_slider, self.size_slider):
            slider.js_property_callbacks.pop("change:value", None)

        style_cb = CustomJS(
            args=dict(
                scatter=self.scatter,
                other_scatters=list(self.other_scatters),
                alpha_slider=self.alpha_slider,
                size_slider=self.size_slider,
            ),
            code="""
            scatter.glyph.size       = size_slider.value;
            scatter.glyph.fill_alpha = alpha_slider.value;
            scatter.change.emit();
            for (let i = 0; i < other_scatters.length; i++) {
                other_scatters[i].glyph.size       = size_slider.value;
                other_scatters[i].glyph.fill_alpha = alpha_slider.value;
            }
            """,
        )
        self.alpha_slider.js_on_change("value", style_cb)
        self.size_slider.js_on_change("value", style_cb)

    def _build_selects(self) -> None:
        """Create X/Y column dropdowns and color-field dropdown. XY callbacks are wired
        later by _wire_xy_selects(), called from _finalize() after real renderers exist."""
        default_color = "gframe_" if "gframe_" in self.columns else self.columns[0]

        self.x_select = Select(title="X", value="x", options=self.columns, width=200)
        self.y_select = Select(title="Y", value="y", options=self.columns, width=200)
        self.color_select = Select(
            title="Color", value=default_color, options=self.columns, width=200
        )

    def _wire_xy_selects(self) -> None:
        """
        Attach XY dropdown callbacks to the *current* self.scatter, self.line, and
        self.source. Must be called from _finalize() after real renderers exist.
        Clears previously attached callbacks first to avoid stacking on re-renders.
        """
        for sel in (self.x_select, self.y_select):
            sel.js_property_callbacks.pop("change:value", None)

        xy_cb = CustomJS(
            args=dict(
                source=self.source,
                scatter=self.scatter,
                line=self.line,
                x_select=self.x_select,
                y_select=self.y_select,
                fig=self.fig,
            ),
            code="""
            const x = x_select.value;
            const y = y_select.value;
            scatter.glyph.x         = { field: x };
            scatter.glyph.y         = { field: y };
            line.glyph.x            = { field: x };
            line.glyph.y            = { field: y };
            fig.xaxis[0].axis_label = x;
            fig.yaxis[0].axis_label = y;
            source.change.emit();
            """,
        )
        self.x_select.js_on_change("value", xy_cb)
        self.y_select.js_on_change("value", xy_cb)

    def _build_raw_checkboxes(self) -> None:
        """
        Create the always-present raw-line / raw-scatter toggle checkboxes.
        These are active in all three view modes. Callbacks are wired later by
        _wire_raw_checkboxes(), called from _finalize() after real renderers exist.
        """
        self.raw_checkboxes = CheckboxButtonGroup(
            labels=["Raw line", "Raw scatter"],
            active=[0, 1],
        )

    def _wire_raw_checkboxes(self) -> None:
        """
        Attach a JS callback to self.raw_checkboxes that toggles the visibility
        of the *current* self.line (index 0) and self.scatter (index 1).
        Clears previously attached callbacks first to avoid stacking on re-renders.
        Must be called from _finalize() after every view method creates real renderers.
        """
        self.raw_checkboxes.js_property_callbacks.pop("change:active", None)

        cb = CustomJS(
            args=dict(line=self.line, scatter=self.scatter),
            code="""
            const active  = new Set(cb_obj.active);
            line.visible    = active.has(0);
            scatter.visible = active.has(1);
            """,
        )
        self.raw_checkboxes.js_on_change("active", cb)

    def _build_checkboxes(self) -> None:
        n = len(self.filtered_columns)
        self.checkboxes: Optional[CheckboxButtonGroup] = (
            CheckboxButtonGroup(
                labels=list(self.filtered_columns.keys()),
                active=list(range(n)),
            )
            if n > 0
            else None
        )

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Assemble the two-column layout from widgets and figures."""
        left = [
            row(self.x_select, self.y_select, self.color_select),
            row(self.alpha_slider, self.size_slider, self.split_slider),
            self.raw_checkboxes,          # always present; wired per view in _finalize
        ]
        if self.checkboxes is not None:
            left.append(self.checkboxes)  # overlay checkboxes; may be disabled per view
        left.append(self.fig)

        # Right column: top spacer accounts for the two slider rows + raw_checkboxes row.
        # An extra spacer is added when overlay checkboxes are present.
        right: list = [Spacer(width=self._figs_width, height=140)]
        if self.checkboxes is not None:
            right.append(Spacer(width=self._figs_width, height=40))
        right.extend(self.side_figs)

        self.layout = row(column(*left), column(*right))

    # ── Shared renderer helpers ───────────────────────────────────────────────

    def _clear_renderers(self) -> None:
        """Remove the current scatter and line from self.fig, if present.
        Also clears legend items so stale entries do not accumulate across view calls.
        """
        for attr in ("scatter", "line"):
            renderer = getattr(self, attr, None)
            if renderer is not None and renderer in self.fig.renderers:
                self.fig.renderers.remove(renderer)
        if self.fig.legend:
            self.fig.legend[0].items = []

    def _make_color_mapper(self, col: str, df=None) -> LinearColorMapper:
        """
        Build a LinearColorMapper over *col* and store it as self.color_mapper.
        Also wires the color_select dropdown to update the mapper live.

        Parameters
        ----------
        col : str  Column name to derive low/high from.
        df  : DataFrame  Defaults to self.df.
        """
        if df is None:
            df = self.df
        self.color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=float(df[col].min()),
            high=float(df[col].max()),
        )
        self._wire_color_select(col)
        return self.color_mapper

    def _wire_color_select(self, initial_col: str) -> None:
        """
        Attach a JS callback to color_select that updates self.color_mapper
        and the scatter glyph's fill colour in real time.
        Called by _make_color_mapper after each mapper is created.
        """
        if self.color_mapper is None:
            return
        cb = CustomJS(
            args=dict(
                glyph=self.scatter.glyph,
                source=self.source,
                mapper=self.color_mapper,
            ),
            code="""
            const field = cb_obj.value;
            const data  = source.data[field];
            if (!data || data.length === 0) return;
            mapper.low        = Math.min(...data);
            mapper.high       = Math.max(...data);
            glyph.fill_color  = { field: field, transform: mapper };
            source.change.emit();
            """,
        )
        # Remove any previous color callback before adding the new one.
        self.color_select.js_property_callbacks.pop("change:value", None)
        self.color_select.js_on_change("value", cb)

    def _render_glyphs(
        self,
        xycols: Sequence[str],
        *,
        line_source,
        scatter_source: ColumnDataSource,
        scatter_color,
        line_alpha: float = 0.4,
        line_color: str = "gray",
        line_width: int = 2,
        scatter_size: float = 1.0,
        scatter_alpha: float = 0.6,
    ) -> None:
        """
        Add a line and scatter to self.fig, storing them as self.line / self.scatter.

        Parameters
        ----------
        line_source : ColumnDataSource  or  tuple[array, array]
            Pass a ColumnDataSource to draw a source-linked line (linked to
            interactive selection), or a (x_array, y_array) tuple to draw a
            static background line directly from array data.
        scatter_source : ColumnDataSource
            Always source-linked; drives interactive selection and hover.
        scatter_color : str  or  bokeh transform
            Flat colour string or a transform(field, mapper) expression.
        """
        x_col, y_col = xycols[0], xycols[1]

        if isinstance(line_source, ColumnDataSource):
            self.line = self.fig.line(
                x=x_col, y=y_col, source=line_source,
                color=line_color, alpha=line_alpha,
                line_width=line_width, legend_label="Path",
            )
        else:
            # Static background line drawn directly from array data.
            x_arr, y_arr = line_source
            self.line = self.fig.line(
                x=x_arr, y=y_arr,
                color=line_color, alpha=line_alpha,
                line_width=line_width, legend_label="Path",
            )

        self.scatter = self.fig.scatter(
            x=x_col, y=y_col, source=scatter_source,
            size=scatter_size, alpha=scatter_alpha,
            color=scatter_color, legend_label="Points",
            nonselection_alpha=0.0,
        )

        # Make the line non-interactive so hover/selection only targets scatter.
        self.line.hover_glyph      = None
        self.line.nonselection_glyph = None
        self.line.level            = "underlay"

    # ── Display ───────────────────────────────────────────────────────────────

    def display(self) -> None:
        """Render self.layout in the current Jupyter output cell."""
        _bokeh_show(self.layout)
