import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    
    def recalculate_pca(self, filtered_data_by_condition):
        """
        Recalculate PCA on filtered data
        
        Args:
            filtered_data_by_condition (dict): Dictionary of filtered DataFrames by condition
        
        Returns:
            dict: Dictionary of DataFrames with updated PCA values
        """
        print("Recalculating PCA for filtered data...")
        
        # Combine all filtered data
        combined_data = pd.concat(filtered_data_by_condition.values()) if filtered_data_by_condition else pd.DataFrame()
        
        if len(combined_data) < 3:
            print("Not enough data points for PCA recalculation.")
            return filtered_data_by_condition
        
        # Check if we have the necessary columns
        required_columns = ['area', 'deformability', 'norm_brightness_q1', 'norm_brightness_q2', 
                           'norm_brightness_q3', 'norm_brightness_q4']
        
        # If normalized brightness columns don't exist, calculate them
        for i in range(1, 5):
            bright_col = f'brightness_q{i}'
            norm_col = f'norm_brightness_q{i}'
            
            if bright_col in combined_data.columns and norm_col not in combined_data.columns:
                combined_data[norm_col] = combined_data.apply(
                    lambda row: row[bright_col] / row['area'] if row['area'] > 0 and pd.notna(row[bright_col]) else 0,
                    axis=1
                )
            elif norm_col not in combined_data.columns:
                # If we don't have brightness data, create zeros
                combined_data[norm_col] = 0
        
        # Now check if we have all required columns
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        if missing_columns:
            print(f"Missing required columns for PCA: {missing_columns}")
            return filtered_data_by_condition
        
        # Prepare data for PCA
        features = ['deformability', 'area', 'norm_brightness_q1', 'norm_brightness_q2', 
                   'norm_brightness_q3', 'norm_brightness_q4']
        X = combined_data[features].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Add PCA results back to the combined data
        combined_data['pca1'] = pca_result[:, 0]
        combined_data['pca2'] = pca_result[:, 1]
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Split back into conditions
        updated_data_by_condition = {}
        for condition, df in filtered_data_by_condition.items():
            # Get indices in the combined dataframe
            condition_indices = combined_data[combined_data['condition'] == condition].index
            
            # Create a copy of the original dataframe
            updated_df = df.copy()
            
            # Update PCA values
            for idx in condition_indices:
                row = combined_data.loc[idx]
                if 'id' in df.columns:
                    # Find the row in the original dataframe by id
                    orig_idx = df[df['id'] == row['id']].index
                    if len(orig_idx) > 0:
                        updated_df.loc[orig_idx[0], 'pca1'] = row['pca1']
                        updated_df.loc[orig_idx[0], 'pca2'] = row['pca2']
                else:
                    # Find by matching area and deformability (less reliable)
                    orig_indices = df[(df['area'] == row['area']) & (df['deformability'] == row['deformability'])].index
                    for orig_idx in orig_indices:
                        updated_df.loc[orig_idx, 'pca1'] = row['pca1']
                        updated_df.loc[orig_idx, 'pca2'] = row['pca2']
            
            updated_data_by_condition[condition] = updated_df
        
        return updated_data_by_condition
    
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
            
            # Create PCA scatter plot if data has PCA components
            has_pca = 'pca1' in all_data.columns and 'pca2' in all_data.columns
            pca_plot = None
            pca_sources = {}
            pca_renderers = {}
            
            if has_pca:
                # Create a new figure for PCA
                pca_plot = figure(
                    width=800, height=600,
                    tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
                    x_axis_label="PC1",
                    y_axis_label="PC2",
                    title="PCA of Cell Morphology and Brightness"
                )
                
                # Style the PCA plot
                pca_plot.title.text_font_size = '16pt'
                pca_plot.xaxis.axis_label_text_font_size = "14pt"
                pca_plot.yaxis.axis_label_text_font_size = "14pt"
                pca_plot.xaxis.major_label_text_font_size = "12pt"
                pca_plot.yaxis.major_label_text_font_size = "12pt"
                pca_plot.grid.grid_line_alpha = 0.3
                pca_plot.outline_line_color = None
                pca_plot.xaxis.axis_line_width = 2
                pca_plot.yaxis.axis_line_width = 2
                
                # Create separate renderers for each condition with unique colors
                for i, (condition, df) in enumerate(data_by_condition.items()):
                    if 'pca1' in df.columns and 'pca2' in df.columns:
                        source = ColumnDataSource(data=df)
                        pca_sources[condition] = source
                        color = palette[i]
                        renderer = pca_plot.scatter(
                            x='pca1', 
                            y='pca2', 
                            source=source,
                            size=8, 
                            fill_color=color,
                            fill_alpha=0.7,
                            line_color=None,
                            legend_label=condition
                        )
                        pca_renderers[condition] = renderer
                
                # Configure legend
                pca_plot.legend.click_policy = "hide"
                pca_plot.legend.location = "top_right"
                
                # Create hover tool for PCA plot
                pca_hover = HoverTool(tooltips=[
                    ("Condition", "@condition"),
                    ("PC1", "@pca1{0.0000}"),
                    ("PC2", "@pca2{0.0000}"),
                    ("Cell Size", "@area{0,0.00} pixels"),
                    ("Deformation", "@deformability{0.0000}")
                ])
                
                if has_ringratio:
                    pca_hover.tooltips.append(("Ring Ratio", "@ringratio{0.0000}"))
                    
                if has_timestamp:
                    pca_hover.tooltips.append(("Time", "@rel_timestamp{0.00} seconds"))
                
                # Add brightness tooltips if available
                for i in range(1, 5):
                    bright_col = f'brightness_q{i}'
                    norm_col = f'norm_brightness_q{i}'
                    
                    if bright_col in all_data.columns:
                        pca_hover.tooltips.append((f"Brightness Q{i}", f"@{bright_col}{{0,0.00}}"))
                    
                    if norm_col in all_data.columns:
                        pca_hover.tooltips.append((f"Norm Brightness Q{i}", f"@{norm_col}{{0.0000}}"))
                
                pca_plot.add_tools(pca_hover)
            
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
            
            # Add PCA recalculation button if PCA data available
            recalculate_pca_button = None
            if has_pca:
                recalculate_pca_button = Button(label="Recalculate PCA", button_type="primary", width=200)
            
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
            
            # Function to update PCA plot with filtered data
            def update_pca_plot():
                if not has_pca:
                    return
                
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
                
                # Update each PCA renderer
                for condition in conditions:
                    if condition in data_by_condition and condition in pca_sources:
                        source = pca_sources[condition]
                        renderer = pca_renderers[condition]
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
                            
                            # Apply timestamp filter if available
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
            
            # Function to recalculate PCA
            def recalculate_pca():
                if not has_pca:
                    return
                
                status_div.text = "<b>Recalculating PCA...</b>"
                doc.add_next_tick_callback(perform_pca_recalculation)
                
            def perform_pca_recalculation():
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
                
                # Recalculate PCA for filtered data
                try:
                    updated_data = self.recalculate_pca(filtered_data_by_condition)
                    
                    # Update data_by_condition with new PCA values
                    for condition, df in updated_data.items():
                        data_by_condition[condition] = df
                    
                    # Update the PCA plot
                    update_pca_plot()
                    
                    status_div.text = "<b>PCA recalculation complete</b>"
                except Exception as e:
                    status_div.text = f"<b>Error recalculating PCA: {e}</b>"
            
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
            
            # Connect PCA recalculation button if available
            if has_pca and recalculate_pca_button:
                recalculate_pca_button.on_click(recalculate_pca)
                
                # Update PCA plot when filters change
                def update_both_plots(attr, old, new):
                    update_plot()
                    update_pca_plot()
                
                # Connect the combined update function to all filters
                condition_select.on_change('active', update_both_plots)
                area_slider.on_change('value', update_both_plots)
                deform_slider.on_change('value', update_both_plots)
                density_slider.on_change('value', update_both_plots)
                opacity_slider.on_change('value', update_both_plots)
                point_size_slider.on_change('value', update_both_plots)
                
                if has_timestamp and timestamp_slider:
                    timestamp_slider.on_change('value', update_both_plots)
                    
                if has_ringratio and ringratio_slider:
                    ringratio_slider.on_change('value', update_both_plots)
            
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
                recalculate_kde_button
            )
            
            # Add PCA recalculation button if available
            if has_pca and recalculate_pca_button:
                controls.children.append(recalculate_pca_button)
                
            controls.children.append(status_div)
            controls.children.append(stats_div)
            
            # Create tabs for scatter plot, histogram, and PCA
            if has_ringratio and has_pca and ringratio_plot and pca_plot:
                try:
                    # First, try importing with newer API
                    from bokeh.models import TabPanel
                    
                    tabs = Tabs(tabs=[
                        TabPanel(child=scatter_plot, title="Scatter Plot"),
                        TabPanel(child=ringratio_plot, title="Ring Ratio Histogram"),
                        TabPanel(child=pca_plot, title="PCA Plot")
                    ])
                except ImportError:
                    # Fallback method for older Bokeh versions
                    try:
                        # Try the older Panel approach
                        tabs = Tabs(tabs=[
                            Panel(child=scatter_plot, title="Scatter Plot"),
                            Panel(child=ringratio_plot, title="Ring Ratio Histogram"),
                            Panel(child=pca_plot, title="PCA Plot")
                        ])
                    except Exception as e:
                        print(f"Error creating tabs: {e}")
                        final_layout = layout([[controls, scatter_plot]])
                        if ringratio_plot and pca_plot:
                            # Still add the plots below as separate panels
                            final_layout = layout([
                                [controls, scatter_plot],
                                [Div(text="<h2>Ring Ratio Histogram</h2>", width=200), ringratio_plot],
                                [Div(text="<h2>PCA Plot</h2>", width=200), pca_plot]
                            ])
                        doc.add_root(final_layout)
                        return
                
                final_layout = layout([[controls, tabs]])
            elif has_ringratio and ringratio_plot:
                try:
                    from bokeh.models import TabPanel
                    
                    tabs = Tabs(tabs=[
                        TabPanel(child=scatter_plot, title="Scatter Plot"),
                        TabPanel(child=ringratio_plot, title="Ring Ratio Histogram")
                    ])
                except ImportError:
                    try:
                        tabs = Tabs(tabs=[
                            Panel(child=scatter_plot, title="Scatter Plot"),
                            Panel(child=ringratio_plot, title="Ring Ratio Histogram")
                        ])
                    except Exception as e:
                        print(f"Error creating tabs: {e}")
                        final_layout = layout([
                            [controls, scatter_plot],
                            [Div(text="<h2>Ring Ratio Histogram</h2>", width=200), ringratio_plot]
                        ])
                        doc.add_root(final_layout)
                        return
                
                final_layout = layout([[controls, tabs]])
            elif has_pca and pca_plot:
                try:
                    from bokeh.models import TabPanel
                    
                    tabs = Tabs(tabs=[
                        TabPanel(child=scatter_plot, title="Scatter Plot"),
                        TabPanel(child=pca_plot, title="PCA Plot")
                    ])
                except ImportError:
                    try:
                        tabs = Tabs(tabs=[
                            Panel(child=scatter_plot, title="Scatter Plot"),
                            Panel(child=pca_plot, title="PCA Plot")
                        ])
                    except Exception as e:
                        print(f"Error creating tabs: {e}")
                        final_layout = layout([
                            [controls, scatter_plot],
                            [Div(text="<h2>PCA Plot</h2>", width=200), pca_plot]
                        ])
                        doc.add_root(final_layout)
                        return
                
                final_layout = layout([[controls, tabs]])
            else:
                final_layout = layout([[controls, scatter_plot]])
            
            # Initialize plots
            update_plot()
            if has_pca:
                update_pca_plot()
            
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
    parser.add_argument('--port', type=int, default=5006, help='Port to run server on (default: 5006)')
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
        print("Initializing plotter...")
        print("Note: If your data contains Brightness_Q1-Q4 columns, they will be normalized by Area")
        print("      and used to calculate PCA components along with Deformability and Area.")
        
        if args.dir:
            plotter = ServerSQLPlotter(data_dir=args.dir, use_gpu=use_gpu)
        else:
            plotter = ServerSQLPlotter(data_path=args.data, db_path=args.db, use_gpu=use_gpu)
        
        # Run in server mode with KDE recalculation
        print("Starting server with KDE and PCA recalculation capability...")
        plotter.run_server(port=args.port)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 