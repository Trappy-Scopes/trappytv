# trappytv2

`trappytv` is visualization and analysis framework for Trappy-Scopes trajectory data, built on top of Bokeh.
It provides the core visualization modes and interactive controls for split, full, and ensemble exploration.

## For Users

`view_ensemble` not working for TrappyTV2.

### Quick Start

```python
from trappytv import TrappyTV

tv = TrappyTV(cell)
tv.view_split(split_no=0)
```

### View Modes

- `view_split(...)`: renders one split with interactive points and linked side plots.
- `view_all(...)`: renders the full trajectory with sampled interactive points.
- `view_ensamble(...)`: renders ensemble trajectories (multiple particles per split) with speed summaries on `fig2`.

### Hover Tooltips

You can auto-build hover tooltips by passing column names:

```python
tv = TrappyTV2(cell, hover_columns=["split", "particle", "frame"])
tv.view_ensamble(hover_columns=["split", "particle", "scopeid", "colony"])
```

You can also update this at runtime:

```python
tv.set_hover_columns(["split", "x", "y", "frame"])
```

## For Developers

### Design Goals

- Consolidate repeated renderer/setup/show logic shared across modes.
- Keep `self.df` as the canonical source without deep copies.
- Separate ensemble data preparation from rendering.

### Internal Structure

- `_clear_main_renderers()`: removes active main plot renderers.
- `_render_main_glyphs(...)`: unified line + scatter creation.
- `_finalize_and_show(...)`: interaction rebuild, hover apply, title, and display.
- `_prepare_ensemble_source(...)`: computes ensemble-ready dataframe, colors, and speed sources.

### Hover Extension Point

- `hover_builder(columns)` receives a sequence of requested columns.
- Default behavior auto-builds tooltip tuples from existing dataframe columns.
- Custom builder can override formatting while retaining per-view `hover_columns` overrides.

### Notes

- API uses `view_split` (instead of `show`) for naming consistency.
- Ensemble API spelling remains `view_ensamble` to match current code.
