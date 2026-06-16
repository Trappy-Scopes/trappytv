# trappytv

Interactive Bokeh visualisation for [Trappy-Scopes](https://trappy-scopes.github.io) trajectory data. Runs inside Jupyter notebooks.

<img src="https://github.com/Trappy-Scopes/trappytv/blob/main/docs/assets/trappytv_logo.png?raw=true">

---

A live demo can be found at: [www.trappy-scopes.github.io/trappytv](https://trappy-scopes.github.io/trappytv).

```python
from trappytv import CellView, TrappyTV

cell = CellView("path/to/data.hd5")
tv   = TrappyTV(cell)
tv.view_split()
```

---

## Installation

```bash
git clone https://github.com/Trappy-Scopes/trappytv.git
```

Requires Python ≥ 3.10, Bokeh ≥ 3.0.

---

## Loading data — `CellView`

`CellView` reads an HDF5 file produced by the Trappy-Scopes postprocessing pipeline. It exposes `cell.dfs["tracks"]` (the trajectory DataFrame) and optionally `cell.fov` (first-frame image), `cell.metadata`, `cell.dfs["xyr"]` (trap-centre trajectory), and `cell.counts`.

```python
cell = CellView(
    "path/to/data.hd5",           # or a postprocess/ directory
    xycols=("x_unrefined", "y_unrefined"),  # columns used for speed computation
    compute_speed=True,            # adds a "speed" column via np.gradient per particle
    verbose=True,                  # print loading stages to stdout
    protected_cols=None,           # extra columns kept by keep_columns()
)
```

Single-cell vs. ensemble is auto-detected from the HDF5 `metadata` node. In ensemble mode `cell.fov` is a `dict[eid, np.ndarray]`; in single-cell mode it is a plain `np.ndarray`.

### Useful methods before visualising

```python
cell.add_columns(inputs, func, outputs)   # derive new columns; func receives the full df
cell.filter(func)                          # row-wise boolean filter: func(df) -> bool Series
cell.keep_columns(cols)                   # drop unused columns (protected cols always kept)
```

`add_columns` behaviour: `inplace=True` (default) means `func` returns a complete replacement DataFrame; `inplace=False` means `func` returns only the new columns, which are merged in by index.

---

## Constructing `TrappyTV`

```python
tv = TrappyTV(
    cell,
    width=1000, height=1000,           # main figure pixel size
    default_xycols=("x_unrefined", "y_unrefined"),
    filtered_columns={                  # overlay trajectories (view_split only)
        "filtered":     ["xf",       "yf"     ],
        "denoised":     ["xf_11Hz",  "yf_11Hz"],
        "anti-aliased": ["xf_3Hz",   "yf_3Hz" ],
    },
    side_cols=[                         # side panels shown to the right
        ("speed",  "Speed"),
        ("signal", "Signal"),
        ("temp",   "Temp"),
    ],
    hover_builder=None,                 # callable(columns) -> list[(label, spec)]
    hover_columns=["split", "frame", "particle"],
)
```

The right-column width equals `width`; side-panel heights are computed automatically so both columns reach the same total pixel height.

### `figs_width` is not a parameter

The right column (histograms + side panels) is always `width` pixels wide. There is no separate `figs_width` argument.

---

## Layout

The widget is a two-column layout:

**Left column** — logo · X/Y/Color dropdowns · Alpha/Size/Split sliders · Raw-glyph and trap/FOV checkboxes · (optional) overlay checkboxes + sliders · main trajectory figure.

**Right column** — text info div · histogram row (5 figures) · per-panel [dropdown + size/alpha sliders + figure] repeated for each side panel.

### Histogram row

Five histogram figures sit above the side panels. The first two show sub-pixel bias (fractional part of `x_unrefined` / `y_unrefined`). Figures 3–5 show residuals between raw and each filtered column (e.g. `xf − x_unrefined`). Histograms update per split alongside the main figure when the split slider moves. Empty histograms are rendered as blank placeholders.

---

## View modes

### `view_split`

```python
tv.view_split(
    split_no=0,           # index into split_values for the initial split
    xycols=None,          # overrides default_xycols
    line_alpha=0.4,
    line_color="gray",
    hover_columns=None,
)
```

Renders one split at a time. The split slider switches between splits live (pure JS). A `full_source` holding all splits is serialised once; the slider filters it client-side and rewrites the display source.

Filtered-column overlays (e.g. denoised, anti-aliased) are drawn as static non-interactive line + scatter glyphs. The checkboxes toggle their visibility. Overlay size and alpha are wired interactively via the overlay sliders.

---

### `view_all`

```python
tv.view_all(
    sample=10,            # scatter shows every Nth row; lower = slower
    xycols=None,
    line_alpha=0.4,
    line_color="gray",
    hover_columns=None,
)
```

Renders the full trajectory across all splits. A static grey line spans every point; the interactive scatter shows every `sample`-th row. The split slider and overlay checkboxes are disabled in this mode. Histograms are computed from the full dataframe.

---

### `view_ensemble`

```python
tv.view_ensemble(
    xycols=("x", "y"),
    smooth_window=25,      # rolling average window for per-particle speed
    exclude_open=False,    # if True, drops rows where trap_open == True
    hover_columns=None,
)
```

Renders all particles across an ensemble. Each `(particle, scopeid)` pair gets a unique colour per split. The split slider switches the ensemble live. The first side panel shows a rolling-averaged speed trace per particle (rendered as `multi_line`). Overlay checkboxes are disabled in this mode. Requires `scopeid` and `particle` columns in the dataframe.

---

## Controls (all modes)

| Control | What it does |
|---|---|
| X / Y dropdowns | Change the main figure axes |
| Color dropdown | Change the colormap source column |
| Alpha / Size sliders | Main scatter appearance |
| Raw line / Raw scatter | Toggle visibility of the primary glyphs |
| Trap boundary / FOV image | Toggle trap-circle and FOV image overlays |
| Split slider | Switch split (live JS in `view_split` and `view_ensemble`) |
| Panel dropdowns | Change the y-column shown in each side panel |
| Panel Size / Alpha | Per-panel scatter appearance (live JS) |
| Overlay checkboxes | Toggle filtered-column overlay visibility (`view_split` only) |
| Overlay Size / Alpha | Overlay glyph appearance (live JS) |

---

## Hover and text info

```python
tv.set_hover_columns(["split", "frame", "particle"])  # replace hover columns live
tv.set_hover_builder(my_fn)                            # replace the tooltip builder
tv.update_text_info("<b>my label</b> | split 3 | 1200 pts")  # update the info div
```

The default hover builder auto-selects `gframe_`, `split`, `x/y`, `frame`, `particle`, `scopeid`, and `dt` (formatted as datetime).

---

## For developers

The codebase is split into two files:

- **`trappycore.py`** — `TrappyCore`: data loading, figure/widget creation, layout assembly, shared renderer helpers. No view logic.
- **`trappytv.py`** — `TrappyTV(TrappyCore)`: hover system, overlay rendering, three view modes, histogram computation.
- **`cellview.py`** — `CellView`: HDF5 reader, speed computation, column helpers.
- **`extraplots.py`** — `subpix_hist`, `residual_hist`: histogram helper functions.

### TrappyCore responsibilities

**Data** — `_load_cell` runs once, renames `gframe → gframe_`, resolves `split_values`. `self.df` is never mutated after this.

**Figures** — `_build_figures` creates `self.fig` (main), `self.hist_figs` (list of 5), and `self.side_figs`. Placeholder invisible renderers (`self.scatter`, `self.line`) are created so wiring methods always have valid targets.

**Widgets** — each widget group has a `_build_*` method. Build methods only create widget objects — no callbacks.

**Wiring** — each `_wire_*` method attaches JS callbacks to the *current* renderers. All wiring clears existing callbacks before adding new ones to prevent stacking across repeated view calls. All wiring runs inside `_finalize`, called at the end of every view method.

**Layout** — `_build_layout` assembles the two-column layout. The right-column top spacer height is `self._controls_h`, derived from module-level `_H_*` constants shared with the `_figs_height` calculation.

**Layout height constants** (tune in `trappycore.py` if alignment is off):

```python
_H_ROW_SELECT   = 60   # selects row
_H_ROW_SLIDER   = 60   # any slider row
_H_ROW_CHECKBOX = 45   # CheckboxButtonGroup row
_H_ROW_OVERLAY  = 60   # overlay checkboxes + sliders row
_H_PANEL_CTRL   = 95   # per-panel control overhead (select + sliders)
_H_TEXT_INFO    = 50   # text info div
_H_HIST         = 140  # histogram row height
```

### TrappyTV view construction

Every view method follows the same sequence:

1. `_clear_renderers()` — removes `self.scatter`, `self.line`, and all overlay renderers; wipes the legend
2. Set `self.source` (and `self.full_source` if a split slider JS callback is needed)
3. `_make_color_mapper(frame_col)` — creates a `LinearColorMapper` (not called in `view_ensemble`)
4. `_render_glyphs(...)` — draws the main line + scatter; sets `self.line` and `self.scatter`
5. `_render_sides(...)` — populates `self.other_scatters`
6. `_precompute_split_hists()` + `_render_hists(...)` — builds and displays histogram row
7. Set `self.frame_col` (used by `_wire_side_selects`)
8. Build a `CustomJS` split callback if needed
9. `_finalize(...)` — wires all callbacks and calls `display()`

### Adding a new view mode

1. Add `view_<name>(self, ...)` following the sequence above.
2. Pass `wire_checkboxes=True` to `_finalize` if overlay checkboxes should be active.
3. Pass `disable_checkboxes=True` if they should be greyed out.
4. The split slider is always present; pass `split_callback=None` to leave it inert, or `disable_split_slider=True` to disable it.
5. When building a split-slider JS callback, pass `split_values=self.split_values` as a CustomJS arg and index it with the slider position: `const split_val = split_values[Math.round(cb_obj.value)]`.

### Hover system

`hover_builder(columns)` is a callable stored on the instance. Override with `set_hover_builder(fn)` or pass `hover_builder=` at construction. `_apply_hover` updates the existing `HoverTool` in place so it always points at the current `self.scatter`.

### Side panel column selector

`_wire_side_selects` passes `yaxis=fig.yaxis[0]` as a direct CustomJS arg (not `fig`) so that `yaxis.axis_label = col` is a reliable direct model update.

### Known bugs

See the issue tracker. Major known issues at time of writing:

- `view_ensemble` split slider JS callback does not use `split_values` for index lookup — produces empty data when split values are not `0, 1, 2, ...`
- `_add_fov` crashes in ensemble mode when `self.fov` is a dict
- `_prepare_ensemble` requires a `scopeid` column; single-cell data will raise `KeyError`
- `_render_sides` background line may crash if `y_col` is missing from `background_df`
- `_wire_xy_selects` passes the whole `fig` object to JS and indexes `fig.xaxis[0]` — axis label updates may be unreliable; use direct axis references as `_wire_side_selects` does
- `view_ensemble` does not update `self.split_slider.value` or `self.split_no` to match the displayed split
