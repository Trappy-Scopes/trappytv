from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Select, HoverTool, CustomJS, ImageURL, LinearColorMapper, LogColorMapper
from bokeh.core.properties import value
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.models import ColorBar
from bokeh.transform import transform
from bokeh.layouts import gridplot, row, column
from bokeh.models import Spacer

output_notebook()



class TrappyTV:
    def __init__(self, cell, width=1000, height=1000, figs_width=140*4, figs_height=80*4):
        
        
        ### Keep a copy of the whole data-frame.
        self.df = cell.dfs["tracks"].rename(columns={"gframe":"gframe_"})
        self.scopeid = cell.scopeid
        
        ## Initial Render is 
        self.source = ColumnDataSource(self.df[self.df.split == 0])
        self.split_no = 0

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
        
        

        ## Plotting objects
        
        # Scatter
        self.color_mapper = linear_cmap(field_name="gframe_", palette=Viridis256, low=self.df["gframe_"].min(), high=self.df["gframe_"].max())
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
        
        
    def __render_interaction__(self):
        
        # ------------------------------------------------------------------
        # Sliders
        # ------------------------------------------------------------------
        self.alpha_slider = Slider(
            start=0.0, end=1.0, value=0.4, step=0.05,
            title="Alpha", width=250
        )

        self.size_slider = Slider(
            start=0.01, end=30, value=0.1, step=0.1,
            title="Size", width=250
        )



        style_callback = CustomJS(
            args=dict(scatter=self.scatter, alpha_slider=self.alpha_slider, size_slider=self.size_slider),
            code="""
            scatter.glyph.size = size_slider.value;
            scatter.glyph.fill_alpha = alpha_slider.value;
            scatter.change.emit();
            """
        )

        self.alpha_slider.js_on_change("value", style_callback)
        self.size_slider.js_on_change("value", style_callback)

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
        
        
        
        self.color_select = Select(title="Color", value="gframe_", options=self.columns)
        self.color_callback = CustomJS(args=dict(glyph=self.scatter.glyph, source=self.source, mapper=self.color_mapper),
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
                """)
        self.color_select.js_on_change("value", self.color_callback)
        
        ## Clear previous hover tools
        self.fig.tools = [t for t in self.fig.tools if not isinstance(t, HoverTool)]
        self.hover = HoverTool(tooltips=[
            ("gframe", "@gframe"),
            ("frame", "@frame"),
            ("split", "@split"),
            ("(X, Y)", "(@x, @y)"),
            ("dt", '@dt{%F %T}')],
            renderers=[self.scatter],
            formatters={'@dt': 'datetime'})
        
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
                                row(self.alpha_slider, self.size_slider),             # sliders
                                self.fig                                              # main figure
                            ),
                            # Right column: fig2 stacked over fig3
                            column(
                                Spacer(width=250*4, height=125),
                                self.fig2,
                                self.fig3,
                                self.fig4
                            )
                        )
        
        #self.color_bar = ColorBar(color_mapper=self.color_mapper, padding=0)
        #self.fig.add_layout(self.color_bar, "right")
    


    def render_sides_all_lines(self):
        self.fig2.title.text = "speed"
        gframe_ = self.df.gframe_
        
        speed_ = self.df["speed"]
        self.fig2.line(x=gframe_,
                       y=speed_,
                       color="gray",
                       alpha=0.2,
                       line_width=2,
                       level="underlay")
        self.fig2.scatter(source=self.source,
                       x="gframe_",
                       y="speed",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)
        
        self.fig3.title.text = "signal"
        signal_ = self.df["signal"]
        self.fig3.line(x=gframe_,
                       y=signal_,
                       color="gray",
                       alpha=0.2,
                       line_width=3,
                       level="underlay")
        self.fig3.scatter(source=self.source,
                       x="gframe_",
                       y="signal",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)
        
        
        self.fig4.title.text = "temp"
        temp_ = self.df["temp"]
        self.fig4.line(x=gframe_,
                       y=temp_,
                       color="gray",
                       alpha=0.2,
                       line_width=3,
                       level="underlay")
        self.fig4.scatter(source=self.source,
                       x="gframe_",
                       y="temp",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)

    
    def render_sides_source(self):
        self.fig2.title.text = "speed"
        self.fig2.line(source=self.source,
                       x="gframe_",
                       y="speed",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.fig2.scatter(source=self.source,
                       x="gframe_",
                       y="speed",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)
        
        self.fig3.title.text = "signal"
        self.fig3.line(source=self.source,
                       x="gframe_",
                       y="signal",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.fig3.scatter(source=self.source,
                       x="gframe_",
                       y="signal",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)
        
        
        self.fig4.title.text = "temp"
        self.fig4.line(source=self.source,
                       x="gframe_",
                       y="temp",
                       color="gray",
                       alpha=0.2,
                       line_width=1)
        self.fig4.scatter(source=self.source,
                       x="gframe_",
                       y="temp",
                       color=transform("gframe_", self.color_mapper),
                       size=3,
                       alpha=0.4,
                       nonselection_alpha=0.0)
    
    
    def show(self, split_no=0, render_circle=False, xycols=["x_unrefined", "y_unrefined"], line_alpha=0.4, line_color="gray"):
        
        self.fig.renderers.remove(self.scatter)
        self.fig.renderers.remove(self.line)
        self.split_no = split_no
        self.source = ColumnDataSource(self.df[self.df.split == split_no])
        
        
        # Line ---- Render lines --> all lines
        self.x_lines = self.df[self.df.split == self.split_no][xycols[0]]
        self.y_lines  = self.df[self.df.split == self.split_no][xycols[1]]
        self.line = self.fig.line(
            self.x_lines, self.y_lines,
            color=line_color,
            alpha=line_alpha,
            line_width=2,
            legend_label="Path"
        )
        
        ## Render scatters
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
        self.source = ColumnDataSource(self.df[::sample])
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
        show(self.layout)
        
        
