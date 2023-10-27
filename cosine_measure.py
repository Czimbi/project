import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def extract_member_from_whole(data, member):
    """Extracts a member recordings from the whole dataset
        Args:
            data (dataframe): The whole dataset.
            member (int): the desired member's number.
        Returns:
            dataframe: Only the desired member's recordins.
    """
    data_cpy = data.copy()

    #Extracting desired member recordings from whole data set
    member_df = data_cpy.loc[data_cpy['Member Id'] == member]

    #Extracting first and last recording date
    date_min = member_df['Time'].min()
    date_max = member_df['Time'].max()

    member_df = member_df.loc[(member_df['Time'] >= date_min) & (member_df['Time'] <= date_max)]

    return member_df
    
def calc_diff_weather(weather_data, weather_attr):
    """Calculates the value difference between two records.
        Args:
            weathe_data (dataframe): 10M weather data.
            weather_attr (str): weather attribute to deal with.
        Returns:
            dataframe: weather_data with 1h difference columnn added
    """
    weather_cpy = weather_data.copy()
    weather_cpy[f'{weather_attr}_shift_-6'] = weather_cpy[weather_attr].shift(6)
    weather_cpy[f'{weather_attr}_diff_1h'] = (weather_cpy[weather_attr] - weather_cpy[f'{weather_attr}_shift_-6']) / weather_cpy[f'{weather_attr}_shift_-6']

    return weather_cpy

def get_shock_dates(weather_data, weather_attr):
    """Extract days where attribute differs from previous at least 30%
        Args:
            weather_data (dataframe): contains weather data in range the member's range. 
            weather_attr (str): the desired weather attribute
        Returns:
            Series: Returns shock dates
    """

    weather_cpy = weather_data.copy()
    
    weather_cpy['date'] = pd.to_datetime(weather_cpy['Time'].dt.date)
    shock_dates = weather_cpy.loc[weather_cpy[f'{weather_attr}_diff_1h'] >= 0.3, 'date']
    
    return shock_dates.unique()

def calc_correlation_shock(data, weather, member, weather_attr, body_attr, day_shift = None):
    """Calculates the correlation between the shock points and biolgical data.

       Args:
            data (dataframe): The whole biological data set
            weather (dataframe): The whole 10M weather data set
            member (int): member's id number
            weather_attr (str): The desired weather attribute to calculate with
            body_atr (str): An attribute name derived from Rmssd
            day_shift (int): day shift
        Returns:
            dataframe: Correlation matrix according to weather and body attribute. 
    """
    data_cpy = data.copy()
    weather_cpy = weather.copy()

    member_df = extract_member_from_whole(data_cpy, member)
    member_df['date'] = pd.to_datetime(member_df['Time'].dt.date)
    date_min = member_df['Time'].min()
    date_max = member_df['Time'].max()
    
    weather_cpy = calc_diff_weather(weather_cpy, weather_attr)
    weather_cpy = weather_cpy.loc[(weather_cpy['Time'] >= date_min) & (weather_cpy['Time'] <= date_max)]

    shock_dates = get_shock_dates(weather_cpy, weather_attr)

    weather_cpy['date'] = pd.to_datetime(weather_cpy['Time'].dt.date)

    weather_cpy = weather_cpy.loc[weather_cpy['date'].isin(shock_dates)]

    merged = pd.merge(member_df, weather_cpy, on='date')
    merged = merged.dropna()

    return merged[[body_attr, weather_attr]].corr()

def scale_weather_data(weather):
    """Scales weather data columns between 0 and 1

        Args:
            weather (dataframe): 1D weather data reduced between member's first and last recoding date
        Returns:
            dataframe: Scaled weather data
    """
    weather_cpy = weather.copy()

    columns = weather_cpy.columns

    scaler = MinMaxScaler()

    weather_scaled = scaler.fit_transform(weather_cpy.to_numpy())

    weather_scaled_df = pd.DataFrame(weather_scaled, columns=columns)

    return weather_scaled_df

def dim_reduce(weather_scaled):
    """Reduces the inputs dimenzions into 2D space.

    Args:
        weather_scaled (dataframe): Scaled (0, 1) weather data
    Returns:
        dataframe: Dimension reduced weather data.
    """
    weather_cpy = weather_scaled.copy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(weather_cpy)
    reduced_df = pd.DataFrame(reduced, columns=['PCA1', 'PCA2'])

    return reduced_df

def plot_similar_hrv(data, weather, member, hrv_attribute):
    """Creates a 2d plot reduced from weather data and labels points with HRV derived value

        Args:
            data (dataframe): whole biological data
            weather (dataframe): whole 1D weather data
            member (int): member that we are interested in
            hrv_attribute (str): the column derived from HRV data
        Returns:
            Figure: scatter plot

    """
    data_cpy = data.copy()
    member_df = extract_member_from_whole(data_cpy, member)
    member_df['date'] = pd.to_datetime(member_df['Time'].dt.date)
    date_min = member_df['date'].min()
    date_max = member_df['date'].max()

    weather_cpy = weather.copy()
    weather_cpy['date'] = pd.to_datetime(weather_cpy['Time'].dt.date)

    weather_cpy = weather_cpy.loc[(weather_cpy['date'] >= date_min) & (weather_cpy['date'] <= date_max)]
    weather_cpy.index = weather_cpy['date']

    scaled = scale_weather_data(weather_cpy.drop(columns=['date', 'index', 'Time'], axis=1))
    reduced = dim_reduce(scaled)
    # reduced['date'] = weather_cpy.index
    reduced['date'] = weather_cpy['date'].values

    merged = pd.merge(reduced, member_df[['date', hrv_attribute]], 'inner', 'date')

    fig = px.scatter(merged, x='PCA1', y='PCA2', color=hrv_attribute, color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_traces(marker_size=10)
    return fig

def impute_data(merged_data, attr):

    merged_cpy = merged_data.copy()
    merged_cpy[f'{attr}_impute'] = merged_cpy[attr].interpolate(method='linear')
    
    return merged_cpy

def calculate_rolling(merged_data, attr):
    
    merged_cpy = merged_data.copy()
    merged_cpy[f'{attr}_rolling'] =  merged_cpy[f'{attr}_impute'].rolling(10).mean()
    return merged_cpy

def shift_rolling(merged_data, attr):

    merged_cpy = merged_data.copy()
    merged_cpy[f'{attr}_shifted'] = merged_cpy[f'{attr}_rolling'].shift(-5)
    return merged_cpy 


def plot_front(data, weather, member, hrv_attribute, weather_attribute):
    """Creates a plot, where the front effects are marked and body value is also ploted. 

    Args:
        data (dataframe): whole data set
        weather (dataframe): whole weather data set
        member (string): member's id
        hrv_attribute (string): hrv or one of ot derivatives
        weather_attribute (string): weather attribute

    Returns:
        Figure: Scatter plot
    """
    data_cpy = data.copy()
    weather_cpy = weather.copy()
    member_df = extract_member_from_whole(data_cpy, member)
    date_min = member_df['Time'].min()
    date_max = member_df['Time'].max()
    member_df['Time'] = pd.to_datetime(member_df['Time'].dt.date)

    weather_cpy = weather_cpy.loc[(weather_cpy['Time'] >= date_min) & (weather_cpy['Time'] <= date_max)]
    #Reading in front data.
    front = pd.read_csv('data/front.csv', sep=';', parse_dates=['Time'])
    front['Time'] = pd.to_datetime(front['Time'], format="%Y.%m.%d", errors="coerce")

    weather_cpy = pd.merge(weather_cpy, front, 'inner', on='Time')

    merged = pd.merge(weather_cpy[['Time', 'code', weather_attribute]], member_df[['Time', hrv_attribute]], 'left', 'Time')
    fig = go.Figure()

    merged = impute_data(merged, hrv_attribute)
    merged = calculate_rolling(merged, hrv_attribute)
    merged = shift_rolling(merged, hrv_attribute)
    merged[f'{hrv_attribute}_is_na'] = merged[hrv_attribute].isna()
    merged[f'{hrv_attribute}_is_na'] = merged[f'{hrv_attribute}_is_na'].replace({True: 'red', False: 'blue'})

    for time, code in zip(merged['Time'], merged['code']):
        if code == 1:
            pass
        elif code == 2:
            fig.add_vline(time, line_color='rgba(255, 0, 0, 0.5)')
        elif code == 3:
            fig.add_vline(time, line_color='rgba(255, 182, 193, 0.5)')
        elif code == 4:
            fig.add_vline(time, line_color='rgba(0, 255, 255, 0.5)')
        elif code == 5:
            fig.add_vline(time, line_color='rgba(0, 0, 255, 0.5)')
        elif code == 6:
            fig.add_vline(time, line_color='rgba(255, 255, 0, 0.5)')

    fig.add_trace(go.Scatter(x=merged['Time'], y=merged[f'{hrv_attribute}_shifted'], mode='lines+markers', marker=dict(color=merged[f'{hrv_attribute}_is_na'])))
            
    return fig
            

#TODO: olyan plottot készíteni, amin a front hatások be vannak húzva és látszóik, a kiválasztott időjárás/testi paraméter


if '__main__' == __name__:
    # """
    #     Reading in weather data and biological data
    # """

    weather_1D = pd.read_csv('data/weather_1D_concat.csv', sep=';', parse_dates=['Time'])
    weather_10M = pd.read_csv('data/weather_10M_concat.csv', sep=';', parse_dates=['Time'])
    whole_data = pd.read_csv('data/hrv_merged.csv', sep=';', parse_dates=['Time'])

    #Creating dropdown list for members
    members = whole_data['Member Id'].unique()
    member_drop_down_list = list(member for member in members)

    #Creating dropdown list for weaather attributes
    weather_dow_down_list = weather_10M.columns.drop(['Time', 'index', 'GroundTempMin'])

    member_drop_down = st.selectbox(
        'Válasszon embert!',
        member_drop_down_list
    )

    weather_drop_down = st.selectbox(
        'Válasszon időjárás paramétert!',
        weather_dow_down_list
    )

    rmssds_drop_down = st.selectbox(
        'Válasszon RMSSD származtatott értéket',
        ['Rmssd', 'Rmssd/RR1', 'Rmssd*RR1']
    )

    # """
    # -------------------------------------------------------------
    #                     End of creating drop downs
    # -------------------------------------------------------------
    # """
    
    corr_matrix = calc_correlation_shock(whole_data, weather_10M, member_drop_down, weather_drop_down, rmssds_drop_down)

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax = ax)
    st.pyplot(fig)

    st.write(f'Időjárás dimenzió csökkentve 2D-ben {rmssds_drop_down}-vel cimkézve.')
    fig_similar = plot_similar_hrv(whole_data, weather_1D, member_drop_down, rmssds_drop_down)

    st.plotly_chart(fig_similar)

    st.write(f'Fronthatások és {rmssds_drop_down} kapcsolata')
    fig_front = plot_front(whole_data, weather_1D, member_drop_down, rmssds_drop_down, weather_drop_down)

    st.plotly_chart(fig_front)



