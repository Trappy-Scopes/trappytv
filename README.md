# trappytv

Interactive Bokeh visualisation for [Trappy-Scopes](https://trappy-scopes.github.io) trajectory data. Runs inside Jupyter notebooks. The visualisation tool is vibecoded using Claude. The analysis pipeline was conceptualised and verified by a human.



<img src="https://github.com/Trappy-Scopes/trappytv/blob/main/docs/assets/trappytv_logo.png?raw=true">

---

A live demo of the tool can be foud at: https://trappy-scopes.github.io/trappytv/

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

`CellView` reads an HDF5 file produced by the Trappy-Scopes postprocessing pipeline and exposes `cell.dfs["tracks"]` to `TrappyTV`.

```python
cell = CellView(
    "path/to/data.hd5",          # or a postprocess/ directory
    xycols=("x_unrefined", "y_unrefined"),
    compute_speed=True,           # adds a "speed" column via gradient
)
```

Useful methods before visualising:

```python
cell.add_columns(inputs, func, outputs)   # derive new columns
cell.filter(func)                          # row-wise boolean filter
cell.keep_columns(cols)                   # drop unused columns (protected cols always kept)
```

---

## Constructing `TrappyTV`

```python
tv = TrappyTV(
    cell,
    width=1000, height=1000,          # main figure pixel size
    figs_width=700,                    # side panel width
    default_xycols=("x_unrefined", "y_unrefined"),
    side_cols=[                        # panels shown to the right
        ("speed",  "Speed"),
        ("signal", "Signal"),
        ("temp",   "Temp"),
    ],
    filtered_columns={                 # overlay trajectories (view_split only)
        "filtered":     ["xf",      "yf"     ],
        "denoised":     ["xf_11Hz", "yf_11Hz"],
        "anti-aliased": ["xf_3Hz",  "yf_3Hz" ],
    },
    hover_columns=["split", "frame", "particle"],
)
```

---

## View modes

### `view_split`

```python
tv.view_split(split_no=0)
```

Renders one split at a time. The **split slider** switches between splits live (pure JS, no Python round-trip). A `full_source` holding all splits is serialised once; the slider filters it client-side and rewrites the display source.

**Filtered-column overlays** (e.g. denoised, anti-aliased trajectories) are drawn as static non-interactive line + scatter glyphs on the main figure. The checkboxes toggle their visibility. Their size and alpha are controlled by the overlay sliders and applied at render time — not interactively — to avoid expensive re-renders.

---

### `view_all`

```python
tv.view_all(sample=10)
```

Renders the full trajectory across all splits. A static grey line spans every point; the interactive scatter shows every `sample`-th row. Useful for getting a whole-experiment overview.

**Split slider and overlay checkboxes are disabled** in this mode. `sample` controls scatter density — lower values are slower to render.

---

### `view_ensemble`

```python
tv.view_ensemble(
    xycols=("x", "y"),
    smooth_window=25,
    exclude_open=False,
)
```

Renders all particles across an ensemble. Each `(particle, scopeid)` pair gets a unique colour per split. The **split slider** switches the ensemble live. The first side panel shows a rolling-averaged speed trace per particle (computed in Python at render time, then serialised as `multi_line` data for JS updates).

**Overlay checkboxes are disabled** in this mode. Does not render filtered-column overlays.

---

## Controls (all modes)

| Control | What it does |
|---|---|
| X / Y dropdowns | Change the main figure axes |
| Color dropdown | Change the colormap source column |
| Alpha / Size sliders | Main scatter only |
| Raw line / Raw scatter | Toggle visibility of the primary glyphs |
| Split slider | Switch split (live JS in `view_split` and `view_ensemble`) |
| Panel dropdowns | Change the y-column shown in each side panel |
| Panel Size / Alpha | Per-panel scatter appearance (live JS) |
| Overlay checkboxes | Toggle filtered-column overlay visibility (`view_split` only) |
| Overlay Size / Alpha | Overlay glyph appearance (applied at next render) |

---

## For developers

The codebase is split into two files:

- **`trappycore.py`** — `TrappyCore`: infrastructure only. No view logic.
- **`trappytv.py`** — `TrappyTV(TrappyCore)`: the three view modes and their helpers.

### TrappyCore responsibilities

**Data** — `_load_cell` runs once, renames `gframe → gframe_`, resolves `split_values`. `self.df` is never mutated after this point.

**Figures** — `_build_figures` creates `self.fig` (main) and `self.side_figs` (list). Placeholder invisible renderers (`self.scatter`, `self.line`) are created so wiring methods always have valid targets before any view is called.

**Widgets** — each widget group has a `_build_*` method (sliders, selects, side selects, side style sliders, raw checkboxes, overlay checkboxes, overlay sliders). Build methods only create the widget objects — no callbacks.

**Wiring** — each `_wire_*` method attaches JS callbacks to the *current* renderers. All wiring methods clear existing callbacks before adding new ones to prevent stacking across repeated view calls. Wiring always runs inside `_finalize`, which is called at the end of every view method.

**Layout** — `_build_layout` assembles the two-column layout. The right-column top spacer height is stored as `self._controls_h`, computed in `__init__` from module-level `_H_*` constants that are shared with the `figs_height` calculation. This keeps the spacer and panel heights permanently in sync.

**Layout height constants** (tune in `trappycore.py` if alignment is off on your display):

```python
_H_ROW_SELECT   = 60   # selects row
_H_ROW_SLIDER   = 60   # any slider row
_H_ROW_CHECKBOX = 45   # CheckboxButtonGroup row
_H_ROW_OVERLAY  = 60   # overlay checkboxes + sliders row
_H_PANEL_CTRL   = 95   # per-panel control overhead (select + sliders)
```

### TrappyTV view construction

Every view method follows the same sequence:

1. `_clear_renderers()` — removes `self.scatter`, `self.line`, and all overlay renderers from `self.fig`; wipes the legend
2. Set `self.source` and (if needed) `self.full_source`
3. `_make_color_mapper(frame_col)` — creates a `LinearColorMapper` and wires `color_select`
4. `_render_glyphs(...)` — draws the main line + scatter; sets `self.line` and `self.scatter`
5. `_render_sides(...)` — populates `self.other_scatters`; reads per-panel slider values at this point
6. Set `self.frame_col` (used by `_wire_side_selects`)
7. Build a `CustomJS` split callback if needed
8. `_finalize(...)` — wires all callbacks to the current renderers and calls `display()`

### Adding a new view mode

1. Add a method `view_<name>(self, ...)` that follows the sequence above.
2. Pass `wire_checkboxes=True` to `_finalize` if overlay checkboxes should be active.
3. Pass `disable_checkboxes=True` if they should be greyed out.
4. The split slider is always present; pass `split_callback=None` to leave it wired but inert, or `disable_split_slider=True` to disable it.

### Hover system

`hover_builder(columns)` is a callable stored on the instance. The default implementation auto-selects useful columns from the dataframe. Override it with `set_hover_builder(fn)` or pass a custom `hover_builder=` at construction. `_apply_hover` updates the existing `HoverTool` in place so it always points at the current `self.scatter`.

### Side panel column selector

`_wire_side_selects` passes `yaxis=fig.yaxis[0]` as a direct `CustomJS` arg (not `fig`) so that `yaxis.axis_label = col` is a guaranteed direct model update — accessing via `fig.yaxis[0]` in JS is unreliable in Bokeh's client-side model graph.