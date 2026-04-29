from copyreg import dispatch_table
from typing import Any


from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Select, HoverTool, CustomJS, ImageURL, LinearColorMapper, LogColorMapper
from bokeh.core.properties import value
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Viridis256
from bokeh.models import ColorBar
from bokeh.transform import transform
from bokeh.layouts import gridplot, row, column
from bokeh.models import Spacer
from bokeh.models import CDSView, ColumnDataSource, GroupFilter





class TrappyTV:

    def get_view_filter(self):
        """Genertaes an appropriate split view filter"""
        self.view_filter = CDSView(filter=GroupFilter(column_name="split", group=self.split_no))
        return self.view_filter

    def __init__(self, cell, width=1000, height=1000, figs_width=140*5, figs_height=None):
        """
        TrappyTV is an interactive Bokeh-based visualization widget for cell tracking data.

        Parameters
        ----------
        cell : object
            A data container object that provides at least a `dfs["tracks"]` DataFrame with a 'gframe' column, 
            and a `scopeid` attribute for labeling.
        width : int, optional
            Width of the main plot in pixels (default: 1000).
        height : int, optional
            Height of the main plot in pixels (default: 1000).
        figs_width : int, optional
            Width of the side plots in pixels (default: 560).
        figs_height : int or None, optional
            Height of the side plots in pixels (default: height/3).

        Attributes
        ----------
        df : pandas.DataFrame
            Internal copy of the tracking data, with 'gframe' renamed to 'gframe_'.
        scopeid : str
            Identifier for the current cell's scope.
        source : bokeh.models.ColumnDataSource
            Data source for the rendered scatter plots, typically for the selected split.
        split_no : int
            Currently displayed split.
        x_init, y_init : str
            Columns to use for initial x and y axes in the main plot.
        fig, fig2, fig3, fig4 : bokeh.plotting.Figure
            Main and side figures for trajectory and feature visualization.
        layout : bokeh.layouts.row
            Complete layout holding all widgets and plots.
        color_mapper : bokeh.models.LinearColorMapper
            Used for coloring scatter points by gframe_.
        logo : bokeh.models.ImageURL
            Custom logo image shown in main plot.

        Notes
        -----
        - The widget allows selection of splits, subsetting, zooming, and inspection of speed/signal/temp over time.
        - Use `show()` to display a single split, or `view_all()` to view the entire trajectory subsampled.
        """
        
        ### Keep a copy of the whole data-frame.
        self.df = cell.dfs["tracks"].rename(columns={"gframe":"gframe_"})
        self.scopeid = cell.scopeid
        
        ## Initial Render is 
        self.source = ColumnDataSource(self.df[self.df.split == 0])
        self.split_no = 0
        self.split_values = sorted(self.df["split"].dropna().unique().tolist()) if "split" in self.df.columns else [0]
        self.view_filter_all_splits = CDSView(filter=GroupFilter(column_name="split", group=self.split_values)) ## Select all splits
        self.view_filter = self.get_view_filter() ## Get 0th view filter.

        # Initial columns --> Default initalisation
        self.x_init = "x_unrefined"
        self.y_init = "y_unrefined"
  
        self.fig = figure(
            width=width,
            height=height,
            title=f"trappytv — split:{self.split_no}",
            tools="box_select, lasso_select, pan, box_zoom, wheel_zoom, reset, save, hover",
            x_axis_label=self.x_init,
            y_axis_label=self.y_init,
            output_backend="webgl"
        )
    
        if figs_height is None:
            figs_height = int(height/3)

        self.fig2 = figure(
            width=figs_width,
            height=figs_height,
            tools="box_select, lasso_select, pan,box_zoom,wheel_zoom,reset,save",
            x_axis_label="gframe_",
            y_axis_label="speed",
            output_backend="webgl"
        )
        
        self.fig3 = figure(
            width=figs_width,
            height=figs_height,
            tools="box_select, lasso_select, pan,box_zoom,wheel_zoom,reset,save",
            x_axis_label="gframe_",
            y_axis_label="signal",
            output_backend="webgl"
        )
        
        self.fig4 = figure(
            width=figs_width,
            height=figs_height,
            tools="box_select, lasso_select, pan,box_zoom,wheel_zoom,reset,save",
            x_axis_label="gframe_",
            y_axis_label="temp",
            output_backend="webgl"
        )
        
        self.other_scatters = [] ## Stores other scatter plots of aux figures.  

        ## Plotting objects
        
        # Scatter
        # Some datasets (e.g. ensemble experiments) may not have `gframe`.
        # Keep the widget functional in that case by skipping gframe-based coloring.
        self.color_mapper = None
        if "gframe_" in self.df.columns:
            self.color_mapper = linear_cmap(
                field_name="gframe_",
                palette=Viridis256,
                low=self.df["gframe_"].min(),
                high=self.df["gframe_"].max(),
            )
        self.scatter = self.fig.scatter(
            self.x_init, self.y_init,
            source=self.source,
            size=1,
            alpha=0.6,
            color="blue",
            legend_label="Points"
        )

        # Line
        self.line = self.fig.line(
            self.x_init, self.y_init,
            source=self.source,
            color="gray",
            alpha=0.4,
            line_width=2,
            legend_label="Path"
        )
        
        self.__render_interaction__()
        
        
    def __render_interaction__(self, frame_col="gframe_"):
        
        # ------------------------------------------------------------------
        # Sliders
        # ------------------------------------------------------------------
        self.alpha_slider = Slider(
            start=0.0, end=1.0, value=0.4, step=0.05,
            title="Alpha", width=250
        )

        self.size_slider = Slider(
            start=0.001, end=20, value=0.1, step=0.1,
            title="Size", width=250
        )

        self.split_slider = Slider(
            start=0,
            end=max(len(self.split_values) - 1, 0),
            value=self.split_values[self.split_no] if self.split_no in self.split_values else 0,
            step=1,
            title="Split no",
            width=250
        )

        style_callback = CustomJS(
            args=dict[str, list[list]](scatter=self.scatter, other_scatters=self.other_scatters, alpha_slider=self.alpha_slider, size_slider=self.size_slider),
            code="""
            scatter.glyph.size = size_slider.value;
            scatter.glyph.fill_alpha = alpha_slider.value;
            scatter.change.emit();
            for (let i = 0; i < other_scatters.length; i++) {
            other_scatters[i].glyph.setv({
                size: size_slider.value,
                fill_alpha: alpha_slider.value})
            }
            """
        )

        self.alpha_slider.js_on_change("value", style_callback)
        self.size_slider.js_on_change("value", style_callback)

        #elf.split_callback = CustomJS(
        #    args=dict(
        #        source=self.source,
        #        fig=self.fig,
        #        split_values=self.split_values,
        #        view=self.get_view_filter()
        #    ),
        #    code="""
        #    const split_idx = Math.round(cb_obj.value);
        #    const split_no = split_values[split_idx];
        #    source.selected.indices = [];
        #    source.change.emit();
        #    fig.title.text = `trappytv — split:${split_no}`;
        #    """
        #)
        #self.split_slider.js_on_change("value", self.split_callback)

        # ------------------------------------------------------------------
        # Dropdowns
        # ------------------------------------------------------------------
        self.columns = list(self.df.columns)

        self.x_select = Select(
            title="X column",
            value=self.x_init,
            options=self.columns,
            width=200
        )

        self.y_select = Select(
            title="Y column",
            value=self.y_init,
            options=self.columns,
            width=200
        )

        self.xy_callback = CustomJS(
            args=dict(
                source=self.source,
                scatter=self.scatter,
                line=self.line,
                x_select=self.x_select,
                y_select=self.y_select,
                p=self.fig
            ),
            code="""
            const x = x_select.value;
            const y = y_select.value;

            scatter.glyph.x = { field: x };
            scatter.glyph.y = { field: y };

            line.glyph.x = { field: x };
            line.glyph.y = { field: y };

            p.xaxis.axis_label = x;
            p.yaxis.axis_label = y;

            source.change.emit();
            """
        )

        self.x_select.js_on_change("value", self.xy_callback)
        self.y_select.js_on_change("value", self.xy_callback)
        
        
        
        # gframe-based coloring is optional (some datasets have no `gframe_`).
        default_color_field = "gframe_" if "gframe_" in self.columns else (self.columns[0] if self.columns else "x")
        self.color_select = Select(
            title="Color",
            value=default_color_field,
            options=self.columns,
            width=200,
            disabled=(self.color_mapper is None),
        )
        if self.color_mapper is not None:
            self.color_callback = CustomJS(
                args=dict(glyph=self.scatter.glyph, source=self.source, mapper=self.color_mapper),
                code="""
                    const field = cb_obj.value;
                    const data = source.data[field];

                    // Compute min/max
                    let min = Math.min(...data);
                    let max = Math.max(...data);

                    // Update mapper
                    mapper.low = min;
                    mapper.high = max;

                    // Update glyph color mapping
                    glyph.fill_color = { field: field, transform: mapper };

                    source.change.emit();
                """,
            )
            self.color_select.js_on_change("value", self.color_callback)
        
        ## Clear previous hover tools
        self.fig.tools = [t for t in self.fig.tools if not isinstance(t, HoverTool)]
        # Hover fields are dataset-dependent.
        hover_tooltips = [("split", "@split"), ("(X, Y)", "(@x, @y)")]
        if "gframe_" in self.columns:
            hover_tooltips.insert(0, ("gframe_", "@gframe_"))
        elif "gframe" in self.columns:
            hover_tooltips.insert(0, ("gframe", "@gframe"))

        if "frame" in self.columns:
            hover_tooltips.append(("frame", "@frame"))
        if "dt" in self.columns:
            hover_tooltips.append(("dt", "@dt{%F %T}"))

        self.hover = HoverTool(
            tooltips=hover_tooltips,
            renderers=[self.scatter],
        )
        
        ## Make the line non-interactive
        self.line.hover_glyph = None
        #self.line.selection_glyph = None
        self.line.nonselection_glyph = None
        self.line.level = "underlay"
        self.fig.add_tools(self.hover)


        ### Lock Aspect Ratio -- Lock all aspect ratios to the same ratio
        for t in self.fig.tools:
            if hasattr(t, "match_aspect"):
                t.match_aspect = True

        self.cont_selection_callback = CustomJS(args=dict(src=self.source, pad=10), code="""
            const inds = [...src.selected.indices].sort((a, b) => a - b);
            if (inds.length === 0) return;

            // --- split into continuous runs ---
            let runs = [];
            let run = [inds[0]];

            for (let i = 1; i < inds.length; i++) {
                if (inds[i] === inds[i - 1] + 1) {
                    run.push(inds[i]);
                } else {
                    runs.push(run);
                    run = [inds[i]];
                }
            }
            runs.push(run);

            // --- keep longest run ---
            let longest = runs.reduce((a, b) => b.length > a.length ? b : a);

            // --- pad selection ---
            const n = src.data.x.length;
            let start = Math.max(0, longest[0] - pad);
            let end   = Math.min(n - 1, longest[longest.length - 1] + pad);

            let new_inds = [];
            for (let i = start; i <= end; i++) {
                new_inds.push(i);
            }

            // --- replace selection ---
            src.selected.indices = new_inds;
            src.change.emit();
        """)

        #self.source.selected.js_on_change("indices", self.cont_selection_callback)
        
        
        
        self.selection_callback = CustomJS(args=dict(source=self.source, figures=[self.fig2, self.fig3, self.fig4]), 
                                      code="""
                                        const inds = source.selected.indices;

                                        if (inds.length === 0) {
                                            for (let i = 0; i < figures.length; i++) {
                                                const datax = source.data['x'];
                                                const datay = source.data['y'];
                                                figures[i].x_range.start = Math.min(...datax);
                                                figures[i].x_range.end   = Math.max(...datax);
                                                figures[i].y_range.start = Math.min(...datay);
                                                figures[i].y_range.end   = Math.max(...datay);
                                            }
                                        } else {
                                            const x_selected = inds.map(i => source.data['x'][i]);
                                            const y_selected = inds.map(i => source.data['y'][i]);
                                            
                                            const min_x = Math.min(...x_selected);
                                            const max_x = Math.max(...x_selected);

                                            const min_y = Math.min(...y_selected);
                                            const max_y = Math.max(...y_selected);
                                            const pad = 0.05 * (max_y - min_y);
                                                

                                            for (let i = 0; i < figures.length; i++) {
                                                figures[i].x_range.start = min_x;
                                                figures[i].x_range.end   = max_x;
                                                figures[i].y_range.start = ymin - pad;
                                                figures[i].y_range.end   = ymax + pad;}
                                            }
                                    """)

        #self.source.selected.js_on_change("indices", self.selection_callback)
        
        
        
        ### Logo
        self.logo = ImageURL(
            url=value("https://github.com/Trappy-Scopes/Trappy-Scopes.github.io/blob/main/docs/assets/tsicon.png?raw=true"),
            w_units="screen",
            h_units="screen",
            x=0,           # pixels from left
            y=1520,        # pixels from bottom
            w=64,           # pixels
            h=25,
            anchor="bottom_left"
        )

        self.fig.add_glyph(self.logo)

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        self.layout = row(
                            # Left column: widgets + main figure
                            column(
                                row(self.x_select, self.y_select, self.color_select),  # dropdowns
                                row(self.alpha_slider, self.size_slider, self.split_slider),             # sliders
                                self.fig                                              # main figure
                            ),
                            # Right column: fig2 stacked over fig3
                            column(
                                Spacer(width=250*4, height=105),
                                self.fig2,
                                self.fig3,
                                self.fig4
                            )
                        )
        
        #self.color_bar = ColorBar(color_mapper=self.color_mapper, padding=0)
        #self.fig.add_layout(self.color_bar, "right")
    


    def render_sides_all_lines(self, frame_col="gframe_"):
        """"""
        self.fig2.title.text = "speed"
        if (frame_col == "gframe_") or (frame_col == "gframe"):
            frame_col_ = self.df["gframe_"]
        else:
            frame_col_ = self.df[frame_col]
        speed_ = self.df["speed"]
        self.other_scatters = []
        self.fig2.line(x=frame_col_,
                       y=speed_,
                       color="gray",
                       alpha=0.2,
                       line_width=2,
                       level="underlay")
        self.other_scatters.append(self.fig2.scatter(source=self.source,
                       x=frame_col,
                       y="speed",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))
        
        self.fig3.title.text = "signal"
        signal_ = self.df["signal"]
        self.fig3.line(x=frame_col_,
                       y=signal_,
                       color="gray",
                       alpha=0.2,
                       line_width=3,
                       level="underlay")
        self.other_scatters.append(self.fig3.scatter(source=self.source,
                       x=frame_col,
                       y="signal",
                       color=transform(frame_col, self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))
        
        
        self.fig4.title.text = "temp"
        temp_ = self.df["temp"]
        self.fig4.line(x=frame_col_,
                       y=temp_,
                       color="gray",
                       alpha=0.2,
                       line_width=3,
                       level="underlay")
        self.other_scatters.append(self.fig4.scatter(source=self.source,
                       x=frame_col,
                       y="temp",
                       color=transform(frame_col, self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))

    
    def render_sides_source(self, frame_col="gframe_"):
        self.fig2.title.text = "speed"
        self.other_scatters = []
        self.fig2.line(source=self.source,
                       x=frame_col,
                       y="speed",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.other_scatters.append(self.fig2.scatter(source=self.source,
                       x=frame_col,
                       y="speed",
                       color=transform(frame_col, self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))
        
        self.fig3.title.text = "signal"
        self.fig3.line(source=self.source,
                       x=frame_col,
                       y="signal",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.other_scatters.append(self.fig3.scatter(source=self.source,
                       x=frame_col,
                       y="signal",
                       color=transform(frame_col, self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))
        
        
        self.fig4.title.text = "temp"
        self.fig4.line(source=self.source,
                       x=frame_col,
                       y="temp",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.other_scatters.append(self.fig4.scatter(source=self.source,
                       x=frame_col,
                       y="temp",
                       color=transform(frame_col, self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0))
        #print(self.other_scatters)
    
    def show(self, split_no=0, render_circle=False, xycols=["x_unrefined", "y_unrefined"], line_alpha=0.4, line_color="gray"):
        
        # Disable the split no slider for this mode
        if hasattr(self, "split_slider"):
            self.split_slider.disabled = True
 
        self.fig.renderers.remove(self.scatter)
        self.fig.renderers.remove(self.line)
        self.split_no = split_no
        self.source.data = ColumnDataSource.from_df(self.df[self.df.split == split_no])
        
        
        # Line ---- Render lines --> all lines
        self.line = self.fig.line(
            source = self.source,
            color=line_color,
            alpha=line_alpha,
            line_width=2,
            legend_label="Path"
        )
        
        ## Render scatters
        self.color_mapper = LinearColorMapper(palette=Viridis256, low=self.df["frame"].min(), high=self.df["frame"].max())
        
        self.scatter = self.fig.scatter(
            xycols[0], xycols[1],
            source=self.source,
            size=1,
            alpha=0.6,
            color=transform("frame", self.color_mapper),#transform("gframe_", self.color_mapper),
            legend_label="Points",
            nonselection_alpha=0.0
        )
        self.render_sides_source()
        self.__render_interaction__()
        show(self.layout)
        
    def view_all(self, sample=10, xycols=["x_unrefined", "y_unrefined"], line_alpha=0.4, line_color="gray"):
        """Render all lines — however, sample every `sample` points.
        This mode is meaningful at sample<15, such that continuous trajectories can be viewed.
        """
        
        ## Clear renderables
        self.fig.renderers.remove(self.scatter)
        self.fig.renderers.remove(self.line)
        

        
        # Line ---- Render lines --> all lines
        self.x_lines = self.df[xycols[0]]
        self.y_lines  = self.df[xycols[1]]
        self.line = self.fig.line(
            self.x_lines, self.y_lines,
            color=line_color,
            alpha=line_alpha,
            line_width=2,
            legend_label="Path"
        )
        
        ## Render scatters
        self.source.data = ColumnDataSource.from_df(self.df[::sample])
        self.color_mapper = LinearColorMapper(palette=Viridis256, low=self.df["gframe_"].min(), high=self.df["gframe_"].max())
        self.scatter = self.fig.scatter(
            xycols[0], xycols[1],
            source=self.source,
            size=1,
            alpha=0.6,
            color=transform("gframe_", self.color_mapper),#transform("gframe_", self.color_mapper),
            legend_label="Points",
            nonselection_alpha=0.0
        )
        self.fig.title.text = f"trappytv :: Cell: {self.scopeid} :: Full-view – Sampling: #{sample}"
        self.__render_interaction__()
        self.render_sides_all_lines()
        #print(self.other_scatters)
        show(self.layout)


    def view_ensamble(self, xycols=["x", "y"], line_alpha=0.4, line_color="gray",
                      smooth_window=25, exclude_open=False):
        """Render all particles in a given ensemble.
        - Colour unique per (particle, scopeid) pair, local to each split
        - Speed calculated from (x, y, frame) with rolling average, plotted on fig2
        - smooth_window  : int  — rolling average window in frames (default 25)
        - exclude_open   : bool — if True, exclude rows where trap_open == True (default False)
        - All markers as circles
        - scopeid + colony in hover tooltip
        - Working split slider updates both fig and fig2
        """
        import numpy as np
        import pandas as pd
        from bokeh.palettes import Category20, Turbo256

        # ── 0. CLEAR RENDERERS + LEGEND ──────────────────────────────────────
        self.fig.renderers.remove(self.scatter)
        self.fig.renderers.remove(self.line)

        if self.fig.legend:
            self.fig.legend[0].items = []

        self.line = self.fig.line(
            x=xycols[0], y=xycols[1],
            source=ColumnDataSource({xycols[0]: [], xycols[1]: []}),
            alpha=0, line_width=0,
        )

        # ── 0b. OPTIONALLY EXCLUDE OPEN TRAPS ────────────────────────────────
        if exclude_open and "trap_open" in self.df.columns:
            self.df = self.df[self.df["trap_open"] != True].copy()

        # ── 1. PALETTE ───────────────────────────────────────────────────────
        max_pairs = (
            self.df[["split", "particle", "scopeid"]]
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
            step    = max(1, 256 // n)
            palette = [Turbo256[i * step] for i in range(n)]

        # ── 2. ASSIGN COLOUR PER (particle, scopeid) PER SPLIT ───────────────
        combos = (
            self.df[["split", "particle", "scopeid"]]
            .drop_duplicates()
            .copy()
        )
        combos["_rank"] = combos.groupby("split", observed=True).cumcount()
        combos["color"] = combos["_rank"].apply(lambda r: palette[r % len(palette)])
        combos = combos.drop(columns="_rank")

        if "color" in self.df.columns:
            self.df = self.df.drop(columns="color")
        self.df = self.df.merge(combos, on=["split", "particle", "scopeid"], how="left")

        # ── 3. CALCULATE SPEED (+ ROLLING AVERAGE) PER SPLIT ─────────────────
        frame_col = "frame" if "frame" in self.df.columns else "gframe_"

        # Keys are STRINGS so JS dict lookup with String(cb_obj.value) matches
        all_speed = {}
        for split_val, grp in self.df.groupby("split", observed=True):
            xs, ys, colors = [], [], []
            for (particle, scopeid), pgrp in grp.groupby(
                ["particle", "scopeid"], sort=False, observed=True
            ):
                pgrp  = pgrp.sort_values(frame_col)
                fx    = pgrp[xycols[0]].to_numpy(dtype=float)
                fy    = pgrp[xycols[1]].to_numpy(dtype=float)
                ft    = pgrp[frame_col].to_numpy(dtype=float)

                dx    = np.diff(fx)
                dy    = np.diff(fy)
                dt    = np.diff(ft)
                dt[dt == 0] = np.nan

                raw_speed = np.sqrt(dx**2 + dy**2) / dt
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

        # ── 4. SOURCES ───────────────────────────────────────────────────────
        self.full_source    = ColumnDataSource(self.df)

        initial_split       = int(self.df["split"].min())
        initial             = self.df[self.df["split"] == initial_split].copy()
        self.display_source = ColumnDataSource(initial)
        self.source         = self.display_source

        init_speed        = all_speed[str(initial_split)]
        self.speed_source = ColumnDataSource(data=dict(
            xs     = init_speed["xs"],
            ys     = init_speed["ys"],
            colors = init_speed["colors"],
        ))

        # ── 5. MAIN SCATTER ──────────────────────────────────────────────────
        self.scatter = self.fig.scatter(
            x      = xycols[0],
            y      = xycols[1],
            source = self.display_source,
            size   = 0.10,
            alpha  = 0.6,
            color  = "color",
            nonselection_alpha = 0.0,
        )

        # ── 6. SPEED LINES ON fig2 ───────────────────────────────────────────
        self.fig2.renderers = []
        self.fig2.multi_line(
            xs         = "xs",
            ys         = "ys",
            source     = self.speed_source,
            line_color = "colors",
            line_width = 1.5,
            alpha      = 0.8,
        )
        self.fig2.xaxis.axis_label = frame_col
        self.fig2.yaxis.axis_label = f"speed (rolling {smooth_window}fr)"
        self.fig2.title.text       = f"Speed per trajectory (window={smooth_window})"

        _empty = ColumnDataSource({"x": [], "y": []})
        self.fig3.renderers = []
        self.fig3.scatter(x="x", y="y", source=_empty, alpha=0)
        self.fig4.renderers = []
        self.fig4.scatter(x="x", y="y", source=_empty, alpha=0)

        # ── 7. REBUILD WIDGETS ───────────────────────────────────────────────
        self.__render_interaction__(frame_col=frame_col)

        # ── 8. PATCH HOVER (add scopeid + colony) ────────────────────────────
        extra_tips = []
        if "scopeid" in self.df.columns:
            extra_tips.append(("scopeid", "@scopeid"))
        if "colony" in self.df.columns:
            extra_tips.append(("colony",  "@colony"))

        if extra_tips:
            for tool in self.fig.tools:
                if isinstance(tool, HoverTool) and self.scatter in tool.renderers:
                    tool.tooltips = tool.tooltips + extra_tips
                    break

        # ── 9. SPLIT SLIDER CALLBACK (after __render_interaction__) ──────────
        callback = CustomJS(
            args=dict(
                display_source = self.display_source,
                full_source    = self.full_source,
                speed_source   = self.speed_source,
                all_speed      = all_speed,
                plot_title     = self.fig.title,
            ),
            code="""
            const split_val = cb_obj.value;
            const split_key = String(split_val);
            const full      = full_source.data;

            // ── Update main scatter ────────────────────────────────────────
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

            // ── Update speed lines ─────────────────────────────────────────
            const sd = all_speed[split_key];
            if (sd !== undefined) {
                speed_source.data = { xs: sd['xs'], ys: sd['ys'], colors: sd['colors'] };
            }

            plot_title.text = 'Trajectories \u2014 split ' + split_val;
            """,
        )

        self.split_slider.js_on_change("value", callback)
        self.fig.title.text = (
            f"trappytv :: Cell: {self.scopeid} :: Ensemble view :: split:: {self.split_no}"
        )
        show(self.layout)