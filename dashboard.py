import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import DualMap
from streamlit_folium import st_folium
import os
import rasterio
from PIL import Image
import numpy as np
import tempfile
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import shutil
import atexit
import re
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Suppress warnings for cleaner Streamlit output
warnings.filterwarnings("ignore")

# ----------------------------- Initialization ----------------------------- #

# Configure Streamlit page
st.set_page_config(
    page_title="Mangrove Analysis Dashboard - Random Forest Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'map_center' not in st.session_state:
    st.session_state.map_center = [-1.9, 105]
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 10
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = True

# Title and Description
st.title("Mangrove Analysis Dashboard - Random Forest Model")
st.markdown("""
This dashboard presents the analysis of mangrove cover and biomass in **Taman Nasional Sembilang**, South Sumatera, Indonesia from **2019 to 2023**.
""")

# Sidebar Controls
st.sidebar.header("Dashboard Controls")

# Define available years
years = [2019, 2020, 2021, 2022, 2023]

# Removed the multiselect for filtering charts
# If you have other controls, you can add them here

# Separate selection for map comparisons
st.sidebar.header("Map Year Selection")
map_year1 = st.sidebar.selectbox(
    "Select Year for Left Map",
    options=years,
    index=0
)
map_year2 = st.sidebar.selectbox(
    "Select Year for Right Map",
    options=years,
    index=len(years)-1
)

# ----------------------------- Helper Functions ----------------------------- #

def create_custom_colormap():
    """Create a custom colormap for visualization."""
    colors_list = ['orange', 'white', 'blue']
    return LinearSegmentedColormap.from_list('custom_diverging', colors_list, N=256)

custom_cmap = create_custom_colormap()

def sanitize_filename(name):
    """Sanitize the layer name to create a valid filename."""
    name = name.replace(' ', '_')
    return re.sub(r'[^\w\-]', '', name)

def convert_geotiff_to_png(geotiff_path, min_val=None, max_val=None, cmap=plt.cm.viridis):
    """Convert a GeoTIFF file to a PNG image."""
    try:
        with rasterio.open(geotiff_path) as src:
            img = src.read(1)
            if img.size <= 1:
                raise ValueError(f"GeoTIFF {geotiff_path} contains insufficient data.")
            
            img = np.where(img == src.nodata, np.nan, img)
            
            if min_val is None:
                min_val = np.nanmin(img)
            if max_val is None:
                max_val = np.nanmax(img)
            
            norm = colors.Normalize(
                vmin=min_val if max_val - min_val != 0 else min_val - 1,
                vmax=max_val if max_val - min_val != 0 else max_val + 1
            )
            
            img_colored = cmap(norm(img))
            img_colored_8bit = (img_colored * 255).astype(np.uint8)
            image = Image.fromarray(img_colored_8bit, mode='RGBA')
            
            return image, src.bounds
    except Exception as e:
        raise Exception(f"Error converting GeoTIFF to PNG: {e}")

def add_image_overlay(map_obj, image, bounds, name, opacity=1):
    """Add an ImageOverlay to the Folium map."""
    try:
        sanitized_name = sanitize_filename(name)
        tmp_path = os.path.join(st.session_state.temp_dir, f"{sanitized_name}.png")
        image.save(tmp_path, format='PNG')
        
        folium_bounds = [
            [bounds.bottom, bounds.left],
            [bounds.top, bounds.right]
        ]
        
        folium.raster_layers.ImageOverlay(
            name=name,
            image=tmp_path,
            bounds=folium_bounds,
            opacity=opacity,
            interactive=True,
            cross_origin=False,
            zindex=1,
        ).add_to(map_obj)
    except Exception as e:
        st.error(f"Error adding image overlay for {name}: {e}")

def create_legend_image(title, min_val, max_val, colors, orientation='horizontal', width=300, height=50):
    """
    Create a gradient legend image.

    Parameters:
    - title (str): Title of the legend.
    - min_val (float): Minimum value.
    - max_val (float): Maximum value.
    - colors (list): List of colors for the gradient.
    - orientation (str): 'horizontal' or 'vertical'.
    - width (int): Width of the image in pixels.
    - height (int): Height of the image in pixels.

    Returns:
    - BytesIO object containing the image.
    """
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    if orientation == 'horizontal':
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        extent = [0, 10, 0, 1]
    else:
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        extent = [0, 1, 0, 10]
    
    cmap = LinearSegmentedColormap.from_list('custom_legend', colors, N=256)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent)
    ax.set_axis_off()
    
    # Add title and labels with white color for min and max
    fig.text(0.5, 0.9, title, ha='center', va='center', fontsize=10, weight='bold', color='black')
    if orientation == 'horizontal':
        fig.text(0.0, 0.5, f"{min_val}", ha='left', va='center', fontsize=8, color='white')
        fig.text(1.0, 0.5, f"{max_val}", ha='right', va='center', fontsize=8, color='white')
    else:
        fig.text(0.5, 0.0, f"{min_val}", ha='center', va='bottom', fontsize=8, color='white')
        fig.text(0.5, 1.0, f"{max_val}", ha='center', va='top', fontsize=8, color='white')
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

# ----------------------------- Sidebar Legends ----------------------------- #

st.sidebar.header("Legends")

# Legend for AGB (C Ton/Ha)
st.sidebar.subheader("AGB (C Ton/Ha)")
agb_colors = ['yellow', 'green', 'darkgreen']
agb_legend = create_legend_image(
    title='',
    min_val=0,
    max_val=300,
    colors=agb_colors,
    orientation='horizontal',
    width=300,
    height=50
)
st.sidebar.image(agb_legend, use_container_width=True)

# Legend for AGB Trend (Ton/year)
st.sidebar.subheader("AGB Trend (Ton/year)")
trend_ton_colors = ['orange', 'white', 'blue']
trend_ton_legend = create_legend_image(
    title='',
    min_val=-10,
    max_val=10,
    colors=trend_ton_colors,
    orientation='horizontal',
    width=300,
    height=50
)
st.sidebar.image(trend_ton_legend, use_container_width=True)

# Legend for AGB Trend (%/year)
st.sidebar.subheader("AGB Trend (%/year)")
trend_percent_colors = ['orange', 'white', 'blue']
trend_percent_legend = create_legend_image(
    title='',
    min_val=-2.5,
    max_val=2.5,
    colors=trend_percent_colors,
    orientation='horizontal',
    width=300,
    height=50
)
st.sidebar.image(trend_percent_legend, use_container_width=True)

# ----------------------------- Load Exported Data ----------------------------- #

@st.cache_data
def load_data(export_folder='GEE_exports'):
    """Load exported CSV data from the specified folder."""
    mangrove_path = os.path.join(export_folder, 'mangrove_area.csv')
    agb_totals_path = os.path.join(export_folder, 'agb_totals.csv')
    feature_importances_path = os.path.join(export_folder, 'feature_importances.csv')
    scatter_plot_path = os.path.join(export_folder, 'scatter_plot_data.csv')
    trend_df_path = os.path.join(export_folder, 'agb_trend.csv')
    # prediction_path is no longer needed since we'll compute it dynamically
    
    # Load CSVs if they exist; otherwise, return empty DataFrames
    mangrove_df = pd.read_csv(mangrove_path) if os.path.exists(mangrove_path) else pd.DataFrame()
    agb_df = pd.read_csv(agb_totals_path) if os.path.exists(agb_totals_path) else pd.DataFrame()
    feature_importances_df = pd.read_csv(feature_importances_path) if os.path.exists(feature_importances_path) else pd.DataFrame()
    scatter_df = pd.read_csv(scatter_plot_path) if os.path.exists(scatter_plot_path) else pd.DataFrame()
    trend_df = pd.read_csv(trend_df_path) if os.path.exists(trend_df_path) else pd.DataFrame()
    # prediction_df will be computed dynamically
    
    return mangrove_df, agb_df, feature_importances_df, scatter_df, trend_df  # Removed prediction_df

# ----------------------------- Initialize Temporary Directory ----------------------------- #

def initialize_temp_dir():
    """Initialize a temporary directory for storing image overlays."""
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

def cleanup_temp_dir():
    """Cleanup the temporary directory upon session end."""
    if 'temp_dir' in st.session_state:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

initialize_temp_dir()
atexit.register(cleanup_temp_dir)

# ----------------------------- Map Visualization Functions ----------------------------- #

def create_agb_comparison_map(center, zoom, export_folder, year1, year2):
    """Create a dual map comparing AGB between two selected years."""
    dual_map = DualMap(
        location=center,
        zoom_start=zoom,
        layout='horizontal'
    )
    
    m1 = dual_map.m1  # Left map
    m2 = dual_map.m2  # Right map
    
    # Add AGB layer for year1 to left map
    agb_path_year1 = os.path.join(export_folder, f'AGB_{year1}.tif')
    if os.path.exists(agb_path_year1):
        try:
            image, bounds = convert_geotiff_to_png(agb_path_year1)
            add_image_overlay(m1, image, bounds, f'AGB {year1} (C Ton/Ha)')
        except Exception as e:
            st.warning(f"Error adding AGB {year1} layer: {e}")
    else:
        st.warning(f"AGB_{year1}.tif not found in the export folder.")
    
    # Add AGB layer for year2 to right map
    agb_path_year2 = os.path.join(export_folder, f'AGB_{year2}.tif')
    if os.path.exists(agb_path_year2):
        try:
            image, bounds = convert_geotiff_to_png(agb_path_year2)
            add_image_overlay(m2, image, bounds, f'AGB {year2} (C Ton/Ha)')
        except Exception as e:
            st.warning(f"Error adding AGB {year2} layer: {e}")
    else:
        st.warning(f"AGB_{year2}.tif not found in the export folder.")
    
    # Add layer controls
    folium.LayerControl().add_to(m1)
    folium.LayerControl().add_to(m2)
    
    return dual_map

def create_trend_comparison_map(center, zoom, export_folder):
    """Create a dual map comparing AGB trends (ton/year and %/year)."""
    dual_map = DualMap(
        location=center,
        zoom_start=zoom,
        layout='horizontal'
    )
    
    m1 = dual_map.m1  # Left map (ton/year)
    m2 = dual_map.m2  # Right map (%/year)
    
    # Add ton/year trend to left map
    trend_ton_path = os.path.join(export_folder, 'AGB_trend_ton_year.tif')
    if os.path.exists(trend_ton_path):
        try:
            image, bounds = convert_geotiff_to_png(
                trend_ton_path,
                min_val=-10,
                max_val=10,
                cmap=custom_cmap
            )
            add_image_overlay(m1, image, bounds, 'AGB Trend (Ton/year)')
            # Legends removed from map
        except Exception as e:
            st.warning(f"Error adding ton/year trend layer: {e}")
    else:
        st.warning("AGB_trend_ton_year.tif not found in the export folder.")
    
    # Add percent/year trend to right map
    trend_percent_path = os.path.join(export_folder, 'AGB_trend_percent_year.tif')
    if os.path.exists(trend_percent_path):
        try:
            image, bounds = convert_geotiff_to_png(
                trend_percent_path,
                min_val=-2.5,
                max_val=2.5,
                cmap=custom_cmap
            )
            add_image_overlay(m2, image, bounds, 'AGB Trend (%/year)')
            # Legends removed from map
        except Exception as e:
            st.warning(f"Error adding percent/year trend layer: {e}")
    else:
        st.warning("AGB_trend_percent_year.tif not found in the export folder.")
    
    # Add layer controls
    folium.LayerControl().add_to(m1)
    folium.LayerControl().add_to(m2)
    
    return dual_map

# ----------------------------- Main Application ----------------------------- #

# Use a spinner for the entire application content
with st.spinner("Loading data and components... Please wait..."):
    # Load data
    export_folder = 'GEE_exports'
    mangrove_df, agb_df, feature_importances_df, scatter_df, trend_df = load_data(export_folder)

    # Initialize containers to hold different sections
    map_section = st.container()
    charts_section = st.container()
    prediction_section = st.container()
    validation_section = st.container()
    feature_importance_section = st.sidebar.container()
    model_performance_section = st.sidebar.container()
    additional_sections = st.container()
    download_section = st.container()

# ----------------------------- Map Visualization Section ----------------------------- #

with map_section:
    st.header("Map Visualizations")
    
    # 1. AGB Comparison Map based on selected years
    st.subheader(f"AGB Comparison: {map_year1} vs {map_year2}")
    try:
        with st.spinner("Generating AGB comparison maps..."):
            agb_dual_map = create_agb_comparison_map(
                st.session_state.map_center,
                st.session_state.zoom_level,
                export_folder,
                map_year1,
                map_year2
            )
        
        with st.spinner("Rendering AGB comparison maps..."):
            map_data = st_folium(
                agb_dual_map,
                width=1280,
                height=720,
                returned_objects=["last_active_drawing", "last_clicked"],
                key="dual_map_agb_comparison"
            )
        
        if map_data is not None and map_data.get('center'):
            st.session_state.map_center = map_data['center']
            st.session_state.zoom_level = map_data.get('zoom', st.session_state.zoom_level)
    
    except Exception as e:
        st.error(f"Error displaying the AGB comparison map: {str(e)}")
    
    # 2. AGB Trend Comparison Map (Fixed years as per data)
    st.subheader("AGB Trend Comparison: Ton/year vs %/year")
    try:
        with st.spinner("Generating AGB trend comparison maps..."):
            trend_dual_map = create_trend_comparison_map(
                st.session_state.map_center,
                st.session_state.zoom_level,
                export_folder
            )
        
        with st.spinner("Rendering AGB trend comparison maps..."):
            map_data_trend = st_folium(
                trend_dual_map,
                width=1280,
                height=720,
                returned_objects=["last_active_drawing", "last_clicked"],
                key="dual_map_trend_comparison"
            )
        
        if map_data_trend is not None and map_data_trend.get('center'):
            st.session_state.map_center = map_data_trend['center']
            st.session_state.zoom_level = map_data_trend.get('zoom', st.session_state.zoom_level)
    
    except Exception as e:
        st.error(f"Error displaying the trend comparison map: {str(e)}")

# ----------------------------- Charts Section ----------------------------- #

with charts_section:
    st.header("Mangrove and Biomass Trends Prediction")
    
    # Create two columns for side-by-side bar charts
    col1, col2 = st.columns(2, gap="large")
    
    # Mangrove Area Chart in first column
    with col1:
        st.subheader("Mangrove Cover Area Over Time")
        if not mangrove_df.empty:
            # No filtering based on selected_years; display all data
            filtered_mangrove_df = mangrove_df
            
            if not filtered_mangrove_df.empty:
                with st.spinner("Generating Mangrove Area chart..."):
                    fig1 = px.bar(
                        filtered_mangrove_df,
                        x='Year',
                        y='Mangrove Area (Ha)',
                        title='Mangrove Cover Area Over Time',
                        labels={'Mangrove Area (Ha)': 'Mangrove Area (Ha)', 'Year': 'Year'},
                        color='Mangrove Area (Ha)',
                        color_continuous_scale='Viridis',
                        template='plotly_white',
                        text_auto=True
                    )
                    fig1.update_layout(
                        title=dict(x=0.5),
                        xaxis=dict(tickmode='linear'),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                with st.spinner("Rendering Mangrove Area chart..."):
                    st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write("No mangrove area data available.")
        else:
            st.write("No mangrove area data available.")
    
    # AGB Chart in second column
    with col2:
        st.subheader("Above-Ground Biomass (AGB) Over Time")
        if not agb_df.empty:
            # No filtering based on selected_years; display all data
            filtered_agb_df = agb_df
            
            if not filtered_agb_df.empty:
                with st.spinner("Generating AGB chart..."):
                    fig2 = px.bar(
                        filtered_agb_df,
                        x='Year',
                        y='AGB (C Ton)',
                        title='Above-Ground Biomass (AGB) Over Time',
                        labels={'AGB (C Ton)': 'AGB (C Ton)', 'Year': 'Year'},
                        color='AGB (C Ton)',
                        color_continuous_scale='Plasma',
                        template='plotly_white',
                        text_auto=True
                    )
                    fig2.update_layout(
                        title=dict(x=0.5),
                        xaxis=dict(tickmode='linear'),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                with st.spinner("Rendering AGB chart..."):
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("No AGB data available.")
        else:
            st.write("No AGB data available.")

# ----------------------------- Prediction Visualization ----------------------------- #

with prediction_section:
    st.header("AGB Prediction for 2024")
    
    # ### Addition: Compute 2024 AGB Prediction
    # Since `agb_prediction_2024.csv` is not available, we'll compute the prediction based on existing AGB data and trend.
    
    if not trend_df.empty and not agb_df.empty:
        try:
            # Extract trend percentage
            trend_percent = trend_df['Trend Percentage (%/year)'].iloc[0]
            
            # Get the latest year's AGB
            latest_year_agb = agb_df['Year'].max()
            agb_latest = agb_df.loc[agb_df['Year'] == latest_year_agb, 'AGB (C Ton)'].values[0]
            
            # Calculate predicted AGB for 2024
            predicted_year_agb = latest_year_agb + 1
            predicted_agb = agb_latest * (1 + trend_percent / 100)
            
            # Create prediction DataFrame
            agb_prediction_df = pd.DataFrame({
                'Year': [predicted_year_agb],
                'Predicted AGB (C Ton)': [predicted_agb]
            })
        except Exception as e:
            st.error(f"Error computing 2024 AGB prediction: {e}")
            agb_prediction_df = pd.DataFrame()
    else:
        st.warning("Insufficient data to compute 2024 AGB prediction.")
        agb_prediction_df = pd.DataFrame()
    
    # ### Addition: Compute 2024 Mangrove Area Prediction
    # We'll use Linear Regression to predict the Mangrove Area for 2024 based on historical data.
    
    if not mangrove_df.empty and len(mangrove_df) >= 2:
        try:
            # Prepare data for regression
            X_mangrove = mangrove_df[['Year']]
            y_mangrove = mangrove_df['Mangrove Area (Ha)']
            
            # Initialize and fit the model
            model_mangrove = LinearRegression()
            model_mangrove.fit(X_mangrove, y_mangrove)
            
            # Predict for 2024
            predicted_year_mangrove = mangrove_df['Year'].max() + 1
            predicted_mangrove_area = model_mangrove.predict([[predicted_year_mangrove]])[0]
            
            # Create prediction DataFrame
            mangrove_prediction_df = pd.DataFrame({
                'Year': [predicted_year_mangrove],
                'Predicted Mangrove Area (Ha)': [predicted_mangrove_area]
            })
        except Exception as e:
            st.error(f"Error computing 2024 Mangrove Area prediction: {e}")
            mangrove_prediction_df = pd.DataFrame()
    else:
        st.warning("Insufficient data to compute 2024 Mangrove Area prediction.")
        mangrove_prediction_df = pd.DataFrame()
    
    # ### Addition: AGB Prediction Visualization
    
    if not agb_prediction_df.empty:
        try:
            predicted_year_agb = agb_prediction_df['Year'].iloc[0]
            predicted_agb = agb_prediction_df['Predicted AGB (C Ton)'].iloc[0]
            
            # Display Predicted AGB as a Metric
            st.subheader(f"Predicted AGB for {predicted_year_agb}")
            st.markdown(f"""
            <div style="background-color:#181d27; padding: 20px; border-radius: 10px; color: white;">
                <h2 style="text-align:center;">{predicted_agb:,.2f} C Ton/Ha</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Combine Historical and Prediction Data for Visualization
            combined_agb_df = agb_df.copy()
            # Replace .append() with pd.concat()
            combined_agb_df = pd.concat([
                combined_agb_df,
                pd.DataFrame({'Year': [predicted_year_agb], 'AGB (C Ton)': [predicted_agb]})
            ], ignore_index=True)
            combined_agb_df = combined_agb_df.sort_values('Year')
            with st.spinner("Generating AGB prediction chart..."):
                fig_prediction = px.line(
                    combined_agb_df,
                    x='Year',
                    y='AGB (C Ton)',
                    title='AGB Trend with 2024 Prediction',
                    markers=True,
                    labels={'AGB (C Ton)': 'AGB (C Ton/Ha)', 'Year': 'Year'},
                    template='plotly_white'
                )
                fig_prediction.add_vline(x=predicted_year_agb, line_dash="dash", line_color="red")
                fig_prediction.add_annotation(
                    x=predicted_year_agb,
                    y=predicted_agb,
                    text="2024 Prediction",
                    showarrow=True,
                    arrowhead=1
                )
                fig_prediction.update_layout(
                    title=dict(x=0.5),
                    xaxis=dict(tickmode='linear'),
                    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            with st.spinner("Rendering AGB prediction chart..."):
                st.plotly_chart(fig_prediction, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying the 2024 AGB prediction: {e}")
    else:
        st.warning("No AGB prediction data available for 2024.")
    
    # ### Addition: Mangrove Area Prediction Visualization
    
    st.header("Mangrove Area Prediction for 2024")
    
    if not mangrove_prediction_df.empty:
        try:
            predicted_year_mangrove = mangrove_prediction_df['Year'].iloc[0]
            predicted_mangrove_area = mangrove_prediction_df['Predicted Mangrove Area (Ha)'].iloc[0]
            
            # Display Predicted Mangrove Area as a Metric
            st.subheader(f"Predicted Mangrove Area for {predicted_year_mangrove}")
            st.markdown(f"""
            <div style="background-color:#181d27; padding: 20px; border-radius: 10px; color: white;">
                <h2 style="text-align:center;">{predicted_mangrove_area:,.2f} Ha</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Combine Historical and Prediction Data for Visualization
            combined_mangrove_df = mangrove_df.copy()
            # Replace .append() with pd.concat()
            combined_mangrove_df = pd.concat([
                combined_mangrove_df,
                pd.DataFrame({'Year': [predicted_year_mangrove], 'Mangrove Area (Ha)': [predicted_mangrove_area]})
            ], ignore_index=True)
            combined_mangrove_df = combined_mangrove_df.sort_values('Year') 
            with st.spinner("Generating Mangrove Area prediction chart..."):
                fig_mangrove_prediction = px.line(
                    combined_mangrove_df,
                    x='Year',
                    y='Mangrove Area (Ha)',
                    title='Mangrove Area Trend with 2024 Prediction',
                    markers=True,
                    labels={'Mangrove Area (Ha)': 'Mangrove Area (Ha)', 'Year': 'Year'},
                    template='plotly_white'
                )
                fig_mangrove_prediction.add_vline(x=predicted_year_mangrove, line_dash="dash", line_color="red")
                fig_mangrove_prediction.add_annotation(
                    x=predicted_year_mangrove,
                    y=predicted_mangrove_area,
                    text="2024 Prediction",
                    showarrow=True,
                    arrowhead=1
                )
                fig_mangrove_prediction.update_layout(
                    title=dict(x=0.5),
                    xaxis=dict(tickmode='linear'),
                    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            with st.spinner("Rendering Mangrove Area prediction chart..."):
                st.plotly_chart(fig_mangrove_prediction, use_container_width=True)        
        except Exception as e:
            st.error(f"Error displaying the 2024 Mangrove Area prediction: {e}")
    else:
        st.warning("No Mangrove Area prediction data available for 2024.")

# ----------------------------- Model Validation ----------------------------- #

with validation_section:
    st.header("Model Validation: AGB Reference vs Prediction")
    if not scatter_df.empty:
        # No filtering based on selected_years; display all data
        filtered_scatter_df = scatter_df
        
        if not filtered_scatter_df.empty:
            with st.spinner("Generating Model Validation scatter plot..."):
                fig_scatter = px.scatter(
                    filtered_scatter_df,
                    x='AGB',
                    y='Prediction',
                    trendline='ols',
                    labels={'AGB': 'Reference AGB (C Ton/Ha)', 'Prediction': 'Predicted AGB (C Ton/Ha)'},
                    title='AGB Reference vs Prediction',
                    template='plotly_white',
                    hover_data=filtered_scatter_df.columns
                )
                
                # Update layout for better aesthetics
                fig_scatter.update_layout(
                    title=dict(x=0.5),
                    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Compute minimum and maximum values across both axes
                min_val = min(filtered_scatter_df['AGB'].min(), filtered_scatter_df['Prediction'].min())
                max_val = max(filtered_scatter_df['AGB'].max(), filtered_scatter_df['Prediction'].max())
                
                # Add 1:1 reference line (Red dashed line)
                fig_scatter.add_shape(
                    type="line",
                    x0=min_val,
                    y0=min_val,
                    x1=max_val,
                    y1=max_val,
                    line=dict(color="Red", dash="dash"),
                )
                
                # Optionally, adjust the axis ranges to add padding
                padding = (max_val - min_val) * 0.05  # 5% padding
                fig_scatter.update_layout(
                    xaxis=dict(range=[min_val - padding, max_val + padding]),
                    yaxis=dict(range=[min_val - padding, max_val + padding]),
                )
            
            with st.spinner("Rendering Model Validation scatter plot..."):
                # Display the scatter plot with the reference line
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.write("No scatter plot data available.")
    else:
        st.write("No scatter plot data available.")

# ----------------------------- Feature Importances and Model Performance ----------------------------- #

with feature_importance_section:
    # Feature Importances
    st.sidebar.subheader("Model Feature Importances")
    if not feature_importances_df.empty:
        with st.spinner("Generating Feature Importances chart..."):
            fig_importance = go.Figure(data=[
                go.Bar(
                    x=feature_importances_df['Importance (%)'],
                    y=feature_importances_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=feature_importances_df['Importance (%)'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
                )
            ])
            fig_importance.update_layout(
                title=dict(text='Feature Importance Distribution', x=0.5, xanchor='center'),
                xaxis_title='Importance (%)',
                yaxis_title='Feature',
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_importance.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128,128,128,0.2)'
            )
        with st.spinner("Rendering Feature Importances chart..."):
            st.sidebar.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.sidebar.write("No feature importances available.")

with model_performance_section:
    # Model R² Score
    st.sidebar.subheader("Model Performance")
    if not scatter_df.empty:
        try:
            filtered_scatter_df = scatter_df
            
            if not filtered_scatter_df.empty:
                with st.spinner("Calculating Model R² Score..."):
                    r2 = r2_score(filtered_scatter_df['AGB'], filtered_scatter_df['Prediction'])
                st.sidebar.metric("Model R² Score", f"{r2:.3f}")
            else:
                st.sidebar.write("Insufficient data to calculate R² score.")
        except Exception as e:
            st.sidebar.error(f"Error calculating R² score: {e}")
    else:
        st.sidebar.write("No scatter plot data available to calculate R² score.")

# ----------------------------- Additional Enhancements ----------------------------- #

with additional_sections:
    # AGB Trend Over Time
    st.subheader("AGB Trend Over Time")
    if not agb_df.empty:
        try:
            with st.spinner("Generating AGB Trend Over Time chart..."):
                # Sort the DataFrame by Year
                agb_df_sorted = agb_df.sort_values('Year').reset_index(drop=True)
                
                # Calculate AGB Change and AGB Change Percentage
                agb_df_sorted['AGB Change'] = agb_df_sorted['AGB (C Ton)'].diff().fillna(0)
                agb_df_sorted['AGB Change (%)'] = agb_df_sorted['AGB (C Ton)'].pct_change().fillna(0) * 100  
        
                # Create the Bar Chart
                fig_combined = px.bar(
                    agb_df_sorted,
                    x='Year',
                    y='AGB Change',
                    title='AGB Trend Over Time',
                    text=agb_df_sorted['AGB Change (%)'].apply(lambda x: f"{x:.2f}%"),
                    labels={'AGB Change': 'AGB Change (C Ton)', 'Year': 'Year'},
                    template='plotly_white',
                    custom_data=['AGB (C Ton)']  # Include AGB (C Ton) in custom data for hover
                )
        
                fig_combined.update_traces(
                    textposition='outside',
                    hovertemplate="<b>Year: %{x}</b><br>AGB Change: %{y} Ton<br>AGB Change (%): %{text}<br>AGB (C Ton): %{customdata[0]:,.2f} Ton<extra></extra>"
                )
        
                # Add Layout Customizations
                fig_combined.update_layout(
                    title=dict(x=0.5),
                    xaxis=dict(tickmode='linear'),
                    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
        
                # **Add Invisible Scatter Trace for Hovering**
                fig_combined.add_trace(
                    go.Scatter(
                        x=agb_df_sorted['Year'],
                        y=agb_df_sorted['AGB Change'],
                        mode='markers',
                        marker=dict(
                            size=20,  # Adjust size as needed
                            color='rgba(0,0,0,0)',  # Fully transparent
                            line=dict(width=0)  # No border
                        ),
                        hoverinfo='text',
                        hovertext=agb_df_sorted['AGB (C Ton)'].apply(lambda x: f"AGB (C Ton): {x:,.2f} Ton"),
                        showlegend=False  # Hide from legend
                    )
                )
        
            with st.spinner("Rendering AGB Trend Over Time chart..."):
                # Display the Combined Chart
                st.plotly_chart(fig_combined, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating AGB Trend Over Time chart: {e}")
    
    
    # ----------------------------- AGB Trend Information ----------------------------- #
    
    st.subheader("AGB Trend Prediction")
    if not trend_df.empty:
        try:
            with st.spinner("Processing AGB Trend Information..."):
                trend_per_year = trend_df['Trend (Ton/year)'].iloc[0]
                trend_percent_per_year = trend_df['Trend Percentage (%/year)'].iloc[0]
                
                # Compute Additional Statistics from agb_df
                if not agb_df.empty and 'Year' in agb_df.columns and 'AGB (C Ton)' in agb_df.columns:
                    agb_df_sorted = agb_df.sort_values('Year')
                    initial_agb = agb_df_sorted['AGB (C Ton)'].iloc[0]
                    final_agb = agb_df_sorted['AGB (C Ton)'].iloc[-1]
                    total_change = final_agb - initial_agb
                    initial_year = agb_df_sorted['Year'].iloc[0]
                    final_year = agb_df_sorted['Year'].iloc[-1]
                else:
                    initial_agb = final_agb = total_change = initial_year = final_year = None
        
        
                # Create a container for the AGB Trend Overview
                with st.container():
                    # Header with Emoji
                    st.markdown("### 📈 AGB Trend Overview")
                    
                    # Create columns for the main trend metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h4 style="text-align:center;">AGB Trend (Ton/year)</h4>
                            <p style="font-size:24px; font-weight:bold; text-align:center;">{trend_per_year:,.3f} Ton/year</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h4 style="text-align:center;">AGB Trend Percentage (%/year)</h4>
                            <p style="font-size:24px; font-weight:bold; text-align:center;">{trend_percent_per_year:.3f}%/year</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional Statistics
                    if all(v is not None for v in [total_change, initial_agb, final_agb, initial_year, final_year]):
                        st.markdown("")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.markdown(f"""
                            <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <p>Total AGB Change</p>
                                <p style="font-size:20px; font-weight:bold;">{total_change:,.2f} Ton</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with stats_col2:
                            st.markdown(f"""
                            <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <p>Initial AGB (Year {initial_year})</p>
                                <p style="font-size:20px; font-weight:bold;">{initial_agb:,.2f} Ton</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with stats_col3:
                            st.markdown(f"""
                            <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <p>Final AGB (Year {final_year})</p>
                                <p style="font-size:20px; font-weight:bold;">{final_agb:,.2f} Ton</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Insufficient data to compute additional statistics.")
                
                #2024
                if not agb_prediction_df.empty and not mangrove_prediction_df.empty:
                    predicted_year_agb = agb_prediction_df['Year'].iloc[0]
                    predicted_agb = agb_prediction_df['Predicted AGB (C Ton)'].iloc[0]
                    predicted_year_mangrove = mangrove_prediction_df['Year'].iloc[0]
                    predicted_mangrove_area = mangrove_prediction_df['Predicted Mangrove Area (Ha)'].iloc[0]
                    
                    st.markdown("")
                    prediction_col1, prediction_col2 = st.columns(2)
                    with prediction_col1:
                        st.markdown(f"""
                        <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h4 style="text-align:center;">Predicted AGB for {predicted_year_agb}</h4>
                            <p style="font-size:20px; font-weight:bold; text-align:center;">{predicted_agb:,.2f} Ton</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with prediction_col2:
                        st.markdown(f"""
                        <div style="background-color:#181d27; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h4 style="text-align:center;">Predicted Mangrove Area for {predicted_year_mangrove}</h4>
                            <p style="font-size:20px; font-weight:bold; text-align:center;">{predicted_mangrove_area:,.2f} Ha</p>
                        </div>
                        """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying AGB trend information: {e}")
    else:
        st.write("Insufficient data to display AGB trend information.")

# ----------------------------- Download Options ----------------------------- #

with download_section:
    st.header("Download Data")
    
    # Download Mangrove Data
    if not mangrove_df.empty:
        with st.spinner("Preparing Mangrove Data for download..."):
            csv_mangrove = mangrove_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Mangrove Data as CSV",
            data=csv_mangrove,
            file_name='mangrove_data.csv',
            mime='text/csv',
        )
    else:
        st.write("No mangrove data available for download.")
    
    # Download AGB Data
    if not agb_df.empty:
        with st.spinner("Preparing AGB Data for download..."):
            csv_agb = agb_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download AGB Data as CSV",
            data=csv_agb,
            file_name='agb_data.csv',
            mime='text/csv',
        )
    else:
        st.write("No AGB data available for download.")
    
    # Download Scatter Plot Data
    if not scatter_df.empty:
        with st.spinner("Preparing Scatter Plot Data for download..."):
            csv_scatter = scatter_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scatter Plot Data as CSV",
            data=csv_scatter,
            file_name='scatter_plot_data.csv',
            mime='text/csv',
        )
    else:
        st.write("No scatter plot data available for download.")
    
    # ### Addition: Download Prediction Data
    # Allow users to download the dynamically generated 2024 prediction data.
    
    if not agb_prediction_df.empty:
        with st.spinner("Preparing 2024 AGB Prediction Data for download..."):
            csv_agb_prediction = agb_prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 2024 AGB Prediction Data as CSV",
            data=csv_agb_prediction,
            file_name='agb_prediction_2024.csv',
            mime='text/csv',
        )
    else:
        st.write("No AGB prediction data available for download.")
    if not mangrove_prediction_df.empty:
        with st.spinner("Preparing 2024 Mangrove Area Prediction Data for download..."):
            csv_mangrove_prediction = mangrove_prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 2024 Mangrove Area Prediction Data as CSV",
            data=csv_mangrove_prediction,
            file_name='mangrove_area_prediction_2024.csv',
            mime='text/csv',
        )
    else:
        st.write("No Mangrove Area prediction data available for download.")

# ----------------------------- Cleanup Temporary Files ----------------------------- #

# Note: Temporary files are managed via st.session_state.temp_dir and cleaned up using atexit.
