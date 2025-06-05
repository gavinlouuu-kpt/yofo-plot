import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import tempfile
import shutil

# Import the SQLPlotter class from sql_plot_mib.py
from sql_plot_mib import SQLPlotter, GPUAcceleratedKDE
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, Button, Div, RangeSlider, CheckboxGroup,
                         Slider, HoverTool, Panel, Tabs, ColorBar, LinearColorMapper)
from bokeh.layouts import column, row, layout
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.palettes import Turbo256, Viridis256

class ServerSQLPlotter(SQLPlotter):
    """
    Extension of SQLPlotter with server mode capability to enable KDE recalculation
    """
    
    def run_server(self, port=5006):
        """
        Run a Bokeh server for interactive plotting with KDE recalculation
        
        Args:
            port (int): Port to run server on
        """
        print(f"Starting Bokeh server on port {port}...")
        
        def make_document(doc):
            """Create a Bokeh document for server-side rendering"""
            # Get data
            data_by_condition = self._get_data_for_plot()
            
            if not data_by_condition:
                print("No data available for plotting.")
                p = figure(width=800, height=600, title="No data available")
                p.text(x=0, y=0, text=['No data available for plotting'], text_font_size='20pt')
                doc.add_root(p)
                return
                
            # Combine all data into a single DataFrame
            all_data = []
            for condition_data in data_by_condition.values():
                all_data.append(condition_data)
            
            if all_data:
                # Use pd.concat instead of deprecated append method
                all_data = pd.concat(all_data) if len(all_data) > 1 else all_data[0]
            else:
                print("No data available for plotting.")
                p = figure(width=800, height=600, title="No data available")
                p.text(x=0, y=0, text=['No data available for plotting'], text_font_size='20pt')
                doc.add_root(p)
                return
            
            # Find global min/max values for scaling
            min_density = all_data['density'].min()
            max_density = all_data['density'].max()
            min_area = all_data['area'].min()
            max_area = all_data['area'].max()
            min_deformability = all_data['deformability'].min()
            max_deformability = all_data['deformability'].max()
            
            # Check for timestamp column
            has_timestamp = 'rel_timestamp' in all_data.columns
            min_timestamp = 0
            max_timestamp = 0
            
            if has_timestamp:
                min_timestamp = 0  # Relative timestamp always starts at 0
                max_timestamp = all_data['rel_timestamp'].max()
                print(f"Found relative timestamp data: range 0 to {max_timestamp:.2f} seconds")
            
            # Check for RingRatio column
            has_ringratio = 'ringratio' in all_data.columns
            min_ringratio = 0
            max_ringratio = 0
            
            if has_ringratio:
                # Clean RingRatio data: Filter out NaN, negative, and zero values
                # Create a filtered copy for RingRatio analysis only
                ringratio_data = all_data.copy()
                ringratio_data = ringratio_data[~ringratio_data['ringratio'].isna()]  # Remove NaN
                ringratio_data = ringratio_data[ringratio_data['ringratio'] > 0]  # Remove <= 0
                
                if len(ringratio_data) == 0:
                    print("Warning: No valid RingRatio values found after filtering. Disabling RingRatio histogram.")
                    has_ringratio = False
                else:
                    print(f"Found {len(ringratio_data)} valid RingRatio values for plotting.")
                    min_ringratio = ringratio_data['ringratio'].min()
                    max_ringratio = ringratio_data['ringratio'].max()
            
            # Create hover tool
            hover = HoverTool(tooltips=[
                ("Condition", "@condition"),
                ("Cell Size", "@area{0,0.00} pixels"),
                ("Deformation", "@deformability{0.0000}"),
                ("Density", "@density{0.00}")
            ])
            
            if has_ringratio:
                hover.tooltips.append(("Ring Ratio", "@ringratio{0.0000}"))
                
            if has_timestamp:
                hover.tooltips.append(("Time", "@rel_timestamp{0.00} seconds"))
            
            # Create figure
            scatter_plot = figure(
                width=800, height=600,
                tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
                x_axis_label="Cell Size (pixels)",
                y_axis_label="Deformation",
                title="Cell Morphology Data"
            )
            
            # Style the plot
            scatter_plot.title.text_font_size = '16pt'
            scatter_plot.xaxis.axis_label_text_font_size = "14pt"
            scatter_plot.yaxis.axis_label_text_font_size = "14pt"
            scatter_plot.xaxis.major_label_text_font_size = "12pt"
            scatter_plot.yaxis.major_label_text_font_size = "12pt"
            scatter_plot.grid.grid_line_alpha = 0.3
            scatter_plot.outline_line_color = None
            scatter_plot.xaxis.axis_line_width = 2
            scatter_plot.yaxis.axis_line_width = 2
            
            # Create separate renderers for each condition with unique colors
            scatter_sources = {}
            scatter_renderers = {}
            
            palette = Turbo256[::max(1, 256 // len(data_by_condition))][:len(data_by_condition)]
            
            for i, (condition, df) in enumerate(data_by_condition.items()):
                source = ColumnDataSource(data=df)
                scatter_sources[condition] = source
                color = palette[i]
                renderer = scatter_plot.scatter(
                    x='area', 
                    y='deformability', 
                    source=source,
                    size=8, 
                    fill_color=color,
                    fill_alpha=0.7,
                    line_color=None,
                    legend_label=condition
                )
                scatter_renderers[condition] = renderer
            
            # Configure legend
            scatter_plot.legend.click_policy = "hide"
            scatter_plot.legend.location = "top_right"
            
            # Create RingRatio histogram plot if data available
            ringratio_plot = None
            hist_sources = {}
            hist_renderers = {}
            
            if has_ringratio:
                # Create a new figure for the histogram
                ringratio_plot = figure(
                    width=800, height=600,
                    tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
                    x_axis_label="Ring Ratio",
                    y_axis_label="Count",
                    title="Ring Ratio Distribution"
                )
                
                # Style the histogram plot
                ringratio_plot.title.text_font_size = '16pt'
                ringratio_plot.xaxis.axis_label_text_font_size = "14pt"
                ringratio_plot.yaxis.axis_label_text_font_size = "14pt"
                ringratio_plot.xaxis.major_label_text_font_size = "12pt"
                ringratio_plot.yaxis.major_label_text_font_size = "12pt"
                ringratio_plot.grid.grid_line_alpha = 0.3
                ringratio_plot.outline_line_color = None
                ringratio_plot.xaxis.axis_line_width = 2
                ringratio_plot.yaxis.axis_line_width = 2
                
                # Create histograms for each condition
                bins = 30  # Default number of bins
                
                for i, (condition, df) in enumerate(data_by_condition.items()):
                    if 'ringratio' in df.columns and not df['ringratio'].isnull().all():
                        # Filter out invalid values for histogram
                        valid_data = df[~df['ringratio'].isna()]  # Remove NaN
                        valid_data = valid_data[valid_data['ringratio'] > 0]  # Remove <= 0
                        
                        if len(valid_data) > 0:
                            # Create histogram data
                            hist, edges = np.histogram(valid_data['ringratio'], 
                                                    bins=bins, 
                                                    range=(min_ringratio, max_ringratio))
                            
                            # Create histogram data source
                            hist_source = ColumnDataSource({
                                'top': hist,
                                'left': edges[:-1],
                                'right': edges[1:],
                                'condition': [condition] * len(hist)
                            })
                            
                            hist_sources[condition] = hist_source
                            
                            # Plot histogram as rectangles
                            color = palette[i]
                            
                            # Create quadrant renderer for the histogram
                            renderer = ringratio_plot.quad(
                                top='top',
                                bottom=0,
                                left='left',
                                right='right',
                                source=hist_source,
                                line_color="white",
                                fill_color=color,
                                fill_alpha=0.7,
                                legend_label=condition
                            )
                            
                            # Store histogram renderer info
                            hist_renderers[condition] = renderer
                            
                # Configure legend for histogram plot
                ringratio_plot.legend.click_policy = "hide"
                ringratio_plot.legend.location = "top_right"
            
            # Create widgets
            
            # Condition selection
            conditions = list(data_by_condition.keys())
            condition_select = CheckboxGroup(
                labels=conditions,
                active=list(range(len(conditions))),
                width=200
            )
            
            # Global filters
            area_slider = RangeSlider(
                title="Cell Size Range (pixels)",
                start=min_area, 
                end=max_area,
                value=(min_area, max_area),
                step=(max_area - min_area) / 100,
                width=400
            )
            
            deform_slider = RangeSlider(
                title="Deformability Range",
                start=min_deformability, 
                end=max_deformability,
                value=(min_deformability, max_deformability),
                step=(max_deformability - min_deformability) / 100,
                width=400
            )
            
            density_slider = RangeSlider(
                title="Density Range",
                start=min_density, 
                end=max_density,
                value=(min_density, max_density),
                step=(max_density - min_density) / 100,
                width=400
            )
            
            # Timestamp slider if data available
            timestamp_slider = None
            if has_timestamp:
                timestamp_slider = RangeSlider(
                    title="Time Range (seconds)",
                    start=min_timestamp, 
                    end=max_timestamp,
                    value=(min_timestamp, max_timestamp),
                    step=(max_timestamp - min_timestamp) / 100 if max_timestamp > min_timestamp else 0.1,
                    width=400
                )
            
            # Create RingRatio slider if data available
            ringratio_slider = None
            if has_ringratio:
                ringratio_slider = RangeSlider(
                    title="Ring Ratio Range",
                    start=min_ringratio, 
                    end=max_ringratio,
                    value=(min_ringratio, max_ringratio),
                    step=(max_ringratio - min_ringratio) / 100,
                    width=400
                )
            
            # Histogram settings
            bins_slider = None
            if has_ringratio:
                bins_slider = Slider(
                    title="Histogram Bins",
                    start=5, 
                    end=50,
                    value=30,
                    step=1,
                    width=400
                )
            
            opacity_slider = Slider(
                title="Point Opacity",
                start=0.1, 
                end=1.0,
                value=0.7,
                step=0.05,
                width=400
            )
            
            point_size_slider = Slider(
                title="Point Size",
                start=2, 
                end=15,
                value=8,
                step=1,
                width=400
            )
            
            # Add KDE recalculation button
            recalculate_kde_button = Button(label="Recalculate KDE", button_type="primary", width=200)
            
            # Stats display
            stats_div = Div(text="", width=400)
            status_div = Div(text="Ready", width=400)
            
            # Function to update the plot with filtered data
            def update_plot():
                # Get selected conditions
                selected_indices = condition_select.active
                selected_conditions = [conditions[i] for i in selected_indices]
                
                # Get filter values
                area_min, area_max = area_slider.value
                deform_min, deform_max = deform_slider.value
                density_min, density_max = density_slider.value
                
                # Additional filter values if available
                timestamp_min, timestamp_max = (timestamp_slider.value if timestamp_slider else (None, None))
                ringratio_min, ringratio_max = (ringratio_slider.value if ringratio_slider else (None, None))
                
                # Stats tracking
                total_visible = 0
                stats = {}
                
                # Update each scatter renderer
                for condition in conditions:
                    if condition in data_by_condition and condition in scatter_sources:
                        source = scatter_sources[condition]
                        renderer = scatter_renderers[condition]
                        df = data_by_condition[condition]
                        
                        # Set visibility
                        renderer.visible = condition in selected_conditions
                        
                        if renderer.visible:
                            # Apply filters
                            mask = (
                                (df['area'] >= area_min) & 
                                (df['area'] <= area_max) &
                                (df['deformability'] >= deform_min) & 
                                (df['deformability'] <= deform_max) &
                                (df['density'] >= density_min) &
                                (df['density'] <= density_max)
                            )
                            
                            # Apply timestamp filter if available - use rel_timestamp which is calculated after loading data
                            if has_timestamp and timestamp_min is not None and timestamp_max is not None and 'rel_timestamp' in df.columns:
                                mask = mask & (df['rel_timestamp'] >= timestamp_min) & (df['rel_timestamp'] <= timestamp_max)
                                
                            # Apply ringratio filter if available
                            if has_ringratio and ringratio_min is not None and ringratio_max is not None:
                                # Only include valid Ring Ratio values (positive numbers)
                                valid_ringratio = (~df['ringratio'].isna()) & (df['ringratio'] > 0)
                                mask = mask & valid_ringratio & (df['ringratio'] >= ringratio_min) & (df['ringratio'] <= ringratio_max)
                            
                            # Filter data
                            filtered_data = df[mask]
                            
                            # Update source with filtered data
                            source.data = dict({col: filtered_data[col].values for col in filtered_data.columns})
                            
                            # Update appearance
                            renderer.glyph.size = point_size_slider.value
                            renderer.glyph.fill_alpha = opacity_slider.value
                            
                            # Store stats
                            stats[condition] = {
                                'visible': len(filtered_data),
                                'total': len(df)
                            }
                            total_visible += len(filtered_data)
                        else:
                            stats[condition] = {
                                'visible': 0,
                                'total': len(df)
                            }
                
                # Update histograms if they exist
                if has_ringratio and ringratio_plot:
                    bin_count = int(bins_slider.value) if bins_slider else 30
                    
                    for condition in conditions:
                        if condition in data_by_condition and condition in hist_renderers:
                            renderer = hist_renderers[condition]
                            source = hist_sources[condition]
                            df = data_by_condition[condition]
                            
                            # Set visibility based on condition selection
                            renderer.visible = condition in selected_conditions
                            
                            if renderer.visible:
                                # Apply filters
                                mask = (
                                    (df['area'] >= area_min) & 
                                    (df['area'] <= area_max) &
                                    (df['deformability'] >= deform_min) & 
                                    (df['deformability'] <= deform_max) &
                                    (df['density'] >= density_min) &
                                    (df['density'] <= density_max) &
                                    (~df['ringratio'].isna()) &
                                    (df['ringratio'] > 0) &
                                    (df['ringratio'] >= ringratio_min) &
                                    (df['ringratio'] <= ringratio_max)
                                )
                                
                                # Apply timestamp filter if available
                                if has_timestamp and timestamp_min is not None and timestamp_max is not None and 'rel_timestamp' in df.columns:
                                    mask = mask & (df['rel_timestamp'] >= timestamp_min) & (df['rel_timestamp'] <= timestamp_max)
                                
                                # Filter data
                                filtered_data = df[mask]
                                
                                if len(filtered_data) > 0:
                                    # Create histogram data
                                    hist, edges = np.histogram(filtered_data['ringratio'], 
                                                           bins=bin_count, 
                                                           range=(min_ringratio, max_ringratio))
                                    
                                    # Update histogram data source
                                    source.data = dict({
                                        'top': hist,
                                        'left': edges[:-1],
                                        'right': edges[1:],
                                        'condition': [condition] * len(hist)
                                    })
                                    
                                    # Update appearance
                                    renderer.glyph.fill_alpha = opacity_slider.value
                
                # Update statistics display
                stats_html = "<h3>Statistics</h3>"
                for condition in selected_conditions:
                    if condition in stats:
                        visible = stats[condition]['visible']
                        total = stats[condition]['total']
                        percentage = total > 0 and (visible / total * 100) or 0
                        stats_html += f"<p><b>{condition}</b>: {visible} cells visible out of {total} total ({percentage:.1f}%)</p>"
                stats_html += f"<p><b>Total visible</b>: {total_visible} cells</p>"
                stats_div.text = stats_html
            
            # Function to recalculate KDE
            def recalculate_kde():
                status_div.text = "<b>Recalculating KDE...</b>"
                doc.add_next_tick_callback(perform_recalculation)
                
            def perform_recalculation():
                # Get selected conditions
                selected_indices = condition_select.active
                selected_conditions = [conditions[i] for i in selected_indices]
                
                # Get filter values
                area_min, area_max = area_slider.value
                deform_min, deform_max = deform_slider.value
                density_min, density_max = density_slider.value
                
                # Additional filter values if available
                timestamp_min, timestamp_max = (timestamp_slider.value if timestamp_slider else (None, None))
                ringratio_min, ringratio_max = (ringratio_slider.value if ringratio_slider else (None, None))
                
                # Build filter dictionary
                filters = {
                    'area': (area_min, area_max),
                    'deformability': (deform_min, deform_max),
                    'density': (density_min, density_max)
                }
                
                # Only add timestamp filter if we're using the actual database column
                # rel_timestamp is calculated after querying and doesn't exist in the database
                if has_timestamp and timestamp_min is not None and timestamp_max is not None:
                    # Check if we should filter by timestamp_us (the database column) instead of rel_timestamp
                    if 'timestamp_us' in all_data.columns:
                        # We'll handle rel_timestamp filtering in memory after getting the data
                        pass
                    
                if has_ringratio and ringratio_min is not None and ringratio_max is not None:
                    filters['ringratio'] = (ringratio_min, ringratio_max)
                
                # Get filtered data
                filtered_data_by_condition = self._get_data_for_plot(
                    conditions=selected_conditions,
                    filters=filters
                )
                
                # Apply timestamp filtering in memory if needed
                if has_timestamp and timestamp_min is not None and timestamp_max is not None:
                    for condition in filtered_data_by_condition:
                        df = filtered_data_by_condition[condition]
                        if 'rel_timestamp' in df.columns:
                            mask = (df['rel_timestamp'] >= timestamp_min) & (df['rel_timestamp'] <= timestamp_max)
                            filtered_data_by_condition[condition] = df[mask]
                
                # Recalculate KDE for each condition
                for condition in selected_conditions:
                    if condition in filtered_data_by_condition:
                        df = filtered_data_by_condition[condition]
                        if len(df) > 5:  # Need enough points for KDE
                            try:
                                # Extract data for KDE calculation
                                points = np.array([[row['area'], row['deformability']] for _, row in df.iterrows()])
                                
                                # Create and initialize KDE
                                kde = GPUAcceleratedKDE(points.T)
                                
                                # Calculate densities
                                densities = kde.evaluate(points.T)
                                
                                # Scale densities to range 0-1
                                min_density = densities.min()
                                max_density = densities.max()
                                if max_density > min_density:
                                    scaled_densities = (densities - min_density) / (max_density - min_density)
                                else:
                                    scaled_densities = densities
                                
                                # Update density values in dataframe
                                df['density'] = scaled_densities
                                
                                # Update the data source
                                if condition in scatter_sources:
                                    source = scatter_sources[condition]
                                    
                                    # Get currently visible data
                                    current_source_data = dict(source.data)
                                    current_ids = set(range(len(current_source_data['area'])))
                                    
                                    # Update densities in the current view
                                    if len(current_ids) > 0:
                                        # Create mapping of area,deformability pairs to new density
                                        point_to_density = {}
                                        for i, (_, row) in enumerate(df.iterrows()):
                                            key = (row['area'], row['deformability'])
                                            point_to_density[key] = scaled_densities[i]
                                        
                                        # Update densities in current view data
                                        new_densities = []
                                        for i in range(len(current_source_data['area'])):
                                            key = (current_source_data['area'][i], current_source_data['deformability'][i])
                                            if key in point_to_density:
                                                new_densities.append(point_to_density[key])
                                            else:
                                                # Keep original density if point not in recalculation set
                                                new_densities.append(current_source_data['density'][i])
                                        
                                        # Update source data with dict constructor as required by Bokeh
                                        current_source_data['density'] = new_densities
                                        source.data = dict(current_source_data)
                                    
                                # Update the stored data
                                data_by_condition[condition] = df
                                print(f"Updated KDE for {condition} with {len(df)} points")
                            except Exception as e:
                                print(f"Error recalculating KDE for condition '{condition}': {e}")
                
                # Update plot
                update_plot()
                status_div.text = "<b>KDE recalculation complete</b>"
            
            # Connect callbacks
            condition_select.on_change('active', lambda attr, old, new: update_plot())
            area_slider.on_change('value', lambda attr, old, new: update_plot())
            deform_slider.on_change('value', lambda attr, old, new: update_plot())
            density_slider.on_change('value', lambda attr, old, new: update_plot())
            opacity_slider.on_change('value', lambda attr, old, new: update_plot())
            point_size_slider.on_change('value', lambda attr, old, new: update_plot())
            
            if has_timestamp and timestamp_slider:
                timestamp_slider.on_change('value', lambda attr, old, new: update_plot())
                
            if has_ringratio and ringratio_slider:
                ringratio_slider.on_change('value', lambda attr, old, new: update_plot())
            
            if has_ringratio and bins_slider:
                bins_slider.on_change('value', lambda attr, old, new: update_plot())
            
            # Connect KDE recalculation button
            recalculate_kde_button.on_click(recalculate_kde)
            
            # Create controls layout
            filters_controls = column(
                Div(text="<h3>Filter Data:</h3>"),
                area_slider,
                deform_slider,
                density_slider
            )
            
            if has_timestamp and timestamp_slider:
                filters_controls.children.append(timestamp_slider)
                
            if has_ringratio and ringratio_slider:
                filters_controls.children.append(ringratio_slider)
            
            appearance_controls = column(
                Div(text="<h3>Appearance:</h3>"),
                opacity_slider,
                point_size_slider
            )
            
            if has_ringratio and bins_slider:
                appearance_controls.children.append(bins_slider)
            
            controls = column(
                Div(text="<h2>Cell Data Visualization</h2>"),
                Div(text="<h3>Select Conditions:</h3>"),
                condition_select,
                filters_controls,
                appearance_controls,
                recalculate_kde_button,
                status_div,
                stats_div
            )
            
            # Create tabs for scatter plot and histogram
            if has_ringratio and ringratio_plot:
                try:
                    # First, try importing with newer API
                    from bokeh.models import TabPanel
                    
                    tabs = Tabs(tabs=[
                        TabPanel(child=scatter_plot, title="Scatter Plot"),
                        TabPanel(child=ringratio_plot, title="Ring Ratio Histogram")
                    ])
                except ImportError:
                    # Fallback method for older Bokeh versions
                    try:
                        # Try the older Panel approach
                        tabs = Tabs(tabs=[
                            Panel(child=scatter_plot, title="Scatter Plot"),
                            Panel(child=ringratio_plot, title="Ring Ratio Histogram")
                        ])
                    except Exception as e:
                        print(f"Error creating tabs: {e}")
                        final_layout = layout([[controls, scatter_plot]])
                        if ringratio_plot:
                            # Still add the histogram below as another panel
                            final_layout = layout([
                                [controls, scatter_plot],
                                [Div(text="<h2>Ring Ratio Histogram</h2>", width=200), ringratio_plot]
                            ])
                        doc.add_root(final_layout)
                        return
                
                final_layout = layout([[controls, tabs]])
            else:
                final_layout = layout([[controls, scatter_plot]])
            
            # Initialize stats
            update_plot()
            
            # Add the layout to the document
            doc.add_root(final_layout)
            doc.title = "Interactive Cell Data Visualization"
        
        # Create a Bokeh application
        app = Application(FunctionHandler(make_document))
        
        # Create and start server
        server = Server({'/': app}, port=port)
        server.start()
        
        # Open in browser
        server.io_loop.add_callback(server.show, "/")
        
        print(f"Server started at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Start the event loop
        server.io_loop.start()


def main():
    """Main function to run the script with server mode"""
    parser = argparse.ArgumentParser(description='Create interactive cell data plots with server-mode KDE recalculation')
    parser.add_argument('--data', type=str, help='Path to CSV or Excel data file')
    parser.add_argument('--dir', type=str, help='Path to directory containing multiple CSV/Excel files to combine')
    parser.add_argument('--db', type=str, help='Path to SQLite database (if data already imported)')
    parser.add_argument('--port', type=int, default=5006, help='Port for Bokeh server (default: 5006)')
    parser.add_argument('--web-port', type=int, default=5000, help='Port for web interface (default: 5000)')
    parser.add_argument('--use-gpu', action='store_true', help='Force GPU usage for calculations')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data and not args.db and not args.dir:
        print("Error: Either --data, --dir, or --db argument is required")
        parser.print_help()
        return
    
    # Determine GPU usage
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    # Create plotter
    try:
        if args.dir:
            plotter = ServerSQLPlotter(data_dir=args.dir, use_gpu=use_gpu)
        else:
            plotter = ServerSQLPlotter(data_path=args.data, db_path=args.db, use_gpu=use_gpu)
        
        # Create Flask app for web interface
        app = Flask(__name__)
        upload_dir = tempfile.mkdtemp()
        
        @app.route('/')
        def index():
            return send_from_directory('.', 'upload.html')
            
        @app.route('/upload', methods=['POST'])
        def upload_files():
            if 'files' not in request.files:
                return jsonify({'success': False, 'message': 'No files uploaded'})
                
            files = request.files.getlist('files')
            if not files:
                return jsonify({'success': False, 'message': 'No files selected'})
                
            # Clear previous uploads
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir)
            
            # Save uploaded files
            filenames = []
            for file in files:
                if file.filename.endswith('.csv'):
                    filepath = os.path.join(upload_dir, file.filename)
                    file.save(filepath)
                    filenames.append(filepath)
            
            if not filenames:
                return jsonify({'success': False, 'message': 'No valid CSV files'})
            
            # Process files and generate plot
            try:
                plotter.load_data(filenames)
                plot_url = f"http://localhost:{args.port}"
                return jsonify({
                    'success': True,
                    'plot_url': plot_url,
                    'message': f'Processed {len(filenames)} files'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error processing files: {str(e)}'
                })
        
        # Start Bokeh server in a separate thread
        from threading import Thread
        bokeh_thread = Thread(target=plotter.run_server, kwargs={'port': args.port})
        bokeh_thread.daemon = True
        bokeh_thread.start()
        
        # Start Flask server
        print(f"Web interface available at http://localhost:{args.web_port}")
        app.run(port=args.web_port)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp files
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()
