# trappytv
**trappytv** is an interactive data visualization tool built on the [Bokeh](https://bokeh.org/) platform for visualizing and exploring cell tracking data from Trappy-Scopes.  
Utilizing Bokeh's Python library, trappytv provides dynamic plots and rich interactive widgets directly in the notebook or web browser, including support for zooming, panning, selection, and visual encoding.

Key features include:
- Interactive trajectory visualization with selection tools
- Linked time series views for features like speed, signal, and temperature
- Customizable views via intuitive sliders and dropdown widgets
- Powered entirely by the Bokeh visualization library for browser-native interactivity

## Description of visulaisation modes:

1. Full view (single particle): view the whole trajectory. Make every `sample` points interactive scatterpoint glyphs. The rest are rendered as a line in the backkground. The main index is the `gframe_` (or gframe). Split no slider interactive tool is disabled during this mode.
2. Split view (single particle): A whole split (`split_no`) is viewed. All points in a split are rendered as a background line and as interactive scatterpoint glyphs. The main index is the `gframe_` (or gframe).
3. Ensemble view (many particles): This mode renders all particles in each split as a background line and as interactive scatterpoint glyphs. The main index is the `frame`. Each point is uniquely determined by the `particle` and `frame` fields.


## Description of interactive elements

+ Row 1:
    1. X column dropdown menu: Select X column for plotting from the datasource.
    2. Y column dropdown menu: Select U column for plotting from the datasource.
    3. Color dropdown menu: Select the column in the datasource that is used for color-coding the rendered trajectory. The Colormap is from the minimum to the maximum value of the selected column.
+ Row 2:
    1. Alpha slider: Adjust the alpha of the interactive scatterplot glyphs.
    2. Size slider: Adjust the size of the interactive scatterplot glyphs.
    3. Split no slider: Select which `split` to render from the datasource.
+ Row 3:
    + Column 1:
            1. Main preview window that shows the trajectory. This is the biggest element of the window and is an interactive plot.
    + Column 2:
        1. Row 1: Plot that shows "speed" coumn from the datasource.
        2. Row 2: Plot that shows "signal" coumn from the datasource.
        3. Row 3: Plot that shows "temp" coumn from the datasource.

## TODO

1. Size and alpha changes do not work on the other scatter plots.
2. Add drop-down for each auxillary plot.
3. X-Y callback does not render underlay lines.