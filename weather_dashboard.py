
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the processed data
df = pd.read_csv('processed_weather_data.csv')  # This would be your processed dataset

# Initialize the Dash app
app = dash.Dash(__name__, title="Global Weather Analysis Dashboard")

# Define the layout
app.layout = html.Div([
    html.H1("Global Weather Analysis Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Top row with filters
    html.Div([
        html.Div([
            html.Label("Select Climate Zone:"),
            dcc.Dropdown(
                id='climate-zone-dropdown',
                options=[{'label': zone, 'value': zone} for zone in df['climate_zone'].unique()],
                value='All',
                clearable=True
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Select Weather Category:"),
            dcc.Dropdown(
                id='weather-dropdown',
                options=[{'label': cat, 'value': cat} for cat in df['weather_category'].unique()],
                value='All',
                clearable=True
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
        
        html.Div([
            html.Label("Temperature Range:"),
            dcc.RangeSlider(
                id='temp-slider',
                min=int(df['temperature_celsius'].min()),
                max=int(df['temperature_celsius'].max()),
                step=1,
                marks={i: f'{i}°C' for i in range(int(df['temperature_celsius'].min()), 
                                                 int(df['temperature_celsius'].max()+1), 5)},
                value=[int(df['temperature_celsius'].min()), int(df['temperature_celsius'].max())]
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30}),
    
    # Main content with visualizations
    html.Div([
        # Left column
        html.Div([
            html.H3("Global Temperature Map", style={'textAlign': 'center'}),
            dcc.Graph(id='temperature-map'),
            
            html.H3("Temperature vs. Humidity by Climate Zone", style={'textAlign': 'center', 'marginTop': 30}),
            dcc.Graph(id='temp-humidity-scatter')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Right column
        html.Div([
            html.H3("Air Quality Distribution", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='air-quality-metric',
                options=[
                    {'label': 'PM2.5', 'value': 'air_quality_PM2.5'},
                    {'label': 'PM10', 'value': 'air_quality_PM10'},
                    {'label': 'Nitrogen Dioxide', 'value': 'air_quality_Nitrogen_dioxide'},
                    {'label': 'Ozone', 'value': 'air_quality_Ozone'},
                    {'label': 'Sulphur Dioxide', 'value': 'air_quality_Sulphur_dioxide'},
                    {'label': 'Carbon Monoxide', 'value': 'air_quality_Carbon_Monoxide'}
                ],
                value='air_quality_PM2.5'
            ),
            dcc.Graph(id='air-quality-box'),
            
            html.H3("Weather Condition Distribution", style={'textAlign': 'center', 'marginTop': 30}),
            dcc.Graph(id='weather-pie')
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    # Bottom row for additional insights
    html.Div([
        html.H3("Climate Cluster Profiles", style={'textAlign': 'center', 'marginTop': 30}),
        dcc.Graph(id='cluster-radar')
    ])
])

# Define callback to update temperature map
@app.callback(
    Output('temperature-map', 'figure'),
    [Input('climate-zone-dropdown', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temp-slider', 'value')]
)
def update_map(climate_zone, weather_category, temp_range):
    filtered_df = df.copy()
    
    # Apply filters
    if climate_zone != 'All' and climate_zone is not None:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    
    if weather_category != 'All' and weather_category is not None:
        filtered_df = filtered_df[filtered_df['weather_category'] == weather_category]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) & 
        (filtered_df['temperature_celsius'] <= temp_range[1])
    ]
    
    # Create the map
    fig = px.scatter_mapbox(
        filtered_df, lat='latitude', lon='longitude', 
        color='temperature_celsius',
        size='temperature_celsius',
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=15,
        zoom=1,
        mapbox_style="carto-positron",
        hover_name='location_name',
        hover_data=['country', 'temperature_celsius', 'humidity', 'condition_text']
    )
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

# Define callback to update temperature-humidity scatter
@app.callback(
    Output('temp-humidity-scatter', 'figure'),
    [Input('climate-zone-dropdown', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temp-slider', 'value')]
)
def update_scatter(climate_zone, weather_category, temp_range):
    filtered_df = df.copy()
    
    # Apply filters
    if climate_zone != 'All' and climate_zone is not None:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    
    if weather_category != 'All' and weather_category is not None:
        filtered_df = filtered_df[filtered_df['weather_category'] == weather_category]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) & 
        (filtered_df['temperature_celsius'] <= temp_range[1])
    ]
    
    # Create the scatter plot
    fig = px.scatter(
        filtered_df, x='temperature_celsius', y='humidity',
        color='climate_zone', size='air_quality_PM2.5',
        hover_name='location_name', hover_data=['country', 'condition_text'],
        opacity=0.7, title="Temperature vs. Humidity"
    )
    
    return fig

# Define callback to update air quality box plot
@app.callback(
    Output('air-quality-box', 'figure'),
    [Input('air-quality-metric', 'value'),
     Input('climate-zone-dropdown', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temp-slider', 'value')]
)
def update_air_quality_box(air_quality_metric, climate_zone, weather_category, temp_range):
    filtered_df = df.copy()
    
    # Apply filters
    if climate_zone != 'All' and climate_zone is not None:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    
    if weather_category != 'All' and weather_category is not None:
        filtered_df = filtered_df[filtered_df['weather_category'] == weather_category]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) & 
        (filtered_df['temperature_celsius'] <= temp_range[1])
    ]
    
    # Create the box plot
    fig = px.box(
        filtered_df, x='climate_zone', y=air_quality_metric,
        color='climate_zone', title=f"{air_quality_metric.replace('air_quality_', '')} by Climate Zone"
    )
    
    return fig

# Define callback to update weather pie chart
@app.callback(
    Output('weather-pie', 'figure'),
    [Input('climate-zone-dropdown', 'value'),
     Input('temp-slider', 'value')]
)
def update_weather_pie(climate_zone, temp_range):
    filtered_df = df.copy()
    
    # Apply filters
    if climate_zone != 'All' and climate_zone is not None:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) & 
        (filtered_df['temperature_celsius'] <= temp_range[1])
    ]
    
    # Count weather categories
    weather_counts = filtered_df['weather_category'].value_counts()
    
    # Create the pie chart
    fig = px.pie(
        names=weather_counts.index, 
        values=weather_counts.values,
        title="Weather Condition Distribution"
    )
    
    return fig

# Define callback to update cluster radar chart
@app.callback(
    Output('cluster-radar', 'figure'),
    [Input('climate-zone-dropdown', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temp-slider', 'value')]
)
def update_cluster_radar(climate_zone, weather_category, temp_range):
    filtered_df = df.copy()
    
    # Apply filters
    if climate_zone != 'All' and climate_zone is not None:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    
    if weather_category != 'All' and weather_category is not None:
        filtered_df = filtered_df[filtered_df['weather_category'] == weather_category]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) & 
        (filtered_df['temperature_celsius'] <= temp_range[1])
    ]
    
    # Define the metrics for the radar chart
    metrics = [
        'temperature_celsius', 'humidity', 'pressure_mb', 
        'wind_kph', 'air_quality_PM2.5', 'uv_index'
    ]
    
    # Create a dataframe with normalized values for each cluster
    cluster_profiles = filtered_df.groupby('cluster')[metrics].mean().reset_index()
    
    # Normalize the data
    for metric in metrics:
        min_val = filtered_df[metric].min()
        max_val = filtered_df[metric].max()
        cluster_profiles[f'{metric}_norm'] = (cluster_profiles[metric] - min_val) / (max_val - min_val)
    
    # Create the radar chart
    fig = go.Figure()
    
    for cluster in cluster_profiles['cluster'].unique():
        cluster_data = cluster_profiles[cluster_profiles['cluster'] == cluster]
        
        fig.add_trace(go.Scatterpolar(
            r=[cluster_data[f'{metric}_norm'].values[0] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Climate Cluster Profiles"
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
