import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pycountry_convert as pc
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDI9byRSHDqVWUaXShHR1WkGTgjlDKwotE")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

st.set_page_config(
    page_title="Covid-19 Dashboard",
    layout="wide"
)

# Load the COVID-19 recovered, confirmed, and death datasets from CSV files
df_recover = pd.read_csv('https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_recovered_v1.csv')
df_death = pd.read_csv('https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_deaths_v1.csv')
df_confirm = pd.read_csv('https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_confirmed_v1.csv')

# Initial preprocessing for recovered data: set first row as header, reset index, and drop old index column
df_recover.columns = df_recover.iloc[0]
df_recover = df_recover[1:].reset_index()
df_recover.drop('index', axis=1, inplace=True)

# Initial preprocessing for death data: set first row as header, reset index, and drop old index column
df_death.columns = df_death.iloc[0]
df_death = df_death[1:].reset_index()
df_death.drop('index', axis=1, inplace=True)

# Melt the confirmed cases dataframe to long format for easier analysis
df_confirm_melt = df_confirm.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
    var_name='Date',
    value_name='Confirm'
)
df_confirm_melt['Date'] = pd.to_datetime(df_confirm_melt['Date'])
df_confirm_melt.set_index('Date', inplace=True)

# Get the minimum and maximum dates from the confirmed cases data
min_date = df_confirm_melt.index.min()
max_date = df_confirm_melt.index.max()

# Helper function to group data by country and date for a given status
def gr(from1, to, df, status, country):
    if country != 'World':
        tem_df = df.loc[to]
    else:
        tem_df = df.loc['2021-5-29']
        grouped = tem_df.groupby('Country/Region', as_index=False).agg({
            status: 'sum',
            'Lat': 'first',
            'Long': 'first'
        }).sort_values('Confirm', ascending=False)
    return grouped

# Melt the recovered cases dataframe to long format
df_recover_melt = df_recover.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
    var_name='Date',
    value_name='Recover'
)
df_recover_melt['Date'] = pd.to_datetime(df_recover_melt['Date'])
df_recover_melt.set_index('Date', inplace=True)

# Melt the death cases dataframe to long format
df_death_melt = df_death.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
    var_name='Date',
    value_name='Death'
)
df_death_melt['Date'] = pd.to_datetime(df_death_melt['Date'])
df_death_melt.set_index('Date', inplace=True)

# Ensure numeric types for main value columns
df_confirm_melt['Confirm'] = df_confirm_melt['Confirm'].astype('float64')
df_death_melt['Death'] = df_death_melt['Death'].astype('float64')
df_recover_melt['Recover'] = df_recover_melt['Recover'].astype('float64')

# Group confirmed cases by date and country, summing up cases and keeping the first Lat/Long for each group
df = df_confirm_melt.groupby([df_confirm_melt.index, 'Country/Region']).agg(
    {
        'Confirm': 'sum',
        'Lat': 'first',
        'Long': 'first',
    }
).reset_index()

# Group recovered cases by date and country, summing up recoveries and keeping the first Lat/Long for each group
df_re = df_recover_melt.groupby([df_recover_melt.index, 'Country/Region']).agg(
    {
        'Recover': 'sum',
        'Lat': 'first',
        'Long': 'first',
    }
).reset_index()

# Group death cases by date and country, summing up deaths and keeping the first Lat/Long for each group
df_de = df_death_melt.groupby([df_death_melt.index, 'Country/Region']).agg(
    {
        'Death': 'sum',
        'Lat': 'first',
        'Long': 'first',
    }
).reset_index()

#  Merge Lat/Long columns from recovered data into confirmed and death dataframes 
df[['Lat', 'Long']] = df_re[['Lat', 'Long']].values
df_de[['Lat', 'Long']] = df_re[['Lat', 'Long']].values

#  Merge confirmed, recovered, and death dataframes on Country/Region, Lat, Long, and Date 
x = df.merge(df_re, on=['Country/Region', 'Lat', 'Long', 'Date'], how='outer')
merge_df = x.merge(df_de, on=['Country/Region', 'Lat', 'Long', 'Date'], how='outer')

#  compute daily and 7-day rolling new cases for each metric 
def compute_features(df):
    df = df.sort_values('Date').copy()
    df['Single_Death'] = df['Death'].diff().fillna(0).clip(lower=0)
    df['Single_Confirm'] = df['Confirm'].diff().fillna(0).clip(lower=0)
    df['Single_Recover'] = df['Recover'].diff().fillna(0).clip(lower=0)
    df['7Day_Death'] = df['Single_Death'].rolling(window=7).mean()
    df['7Day_Confirm'] = df['Single_Confirm'].rolling(window=7).mean()
    df['7Day_Recover'] = df['Single_Recover'].rolling(window=7).mean()
    return df

#  Apply feature engineering to each country and reset the index 
complete_dataset = merge_df.groupby('Country/Region', group_keys=False).apply(compute_features).reset_index(drop=True)
complete_dataset[['Lat', 'Long']] = complete_dataset[['Lat', 'Long']].astype('float64')
complete_dataset['Date'] = pd.to_datetime(complete_dataset['Date'])

#  Standardise country names and import continent mapping library 
complete_dataset['Country/Region'] = complete_dataset['Country/Region'].str.replace('US', 'USA')
df_confirm_melt['Country/Region'] = df_confirm_melt['Country/Region'].str.replace('US', 'USA')

#  Helper function: map country name to continent name 
def country_to_continent(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        # Map continent code to name
        continent_name = {
            'AF': 'Africa',
            'AS': 'Asia',
            'EU': 'Europe',
            'NA': 'North America',
            'SA': 'South America',
            'OC': 'Oceania',
            'AN': 'Antarctica'
        }[continent_code]
        return continent_name
    except:
        return 'Unknown'

#  Assign continent to each country in main and melted confirmed dataframes 
complete_dataset['Continent'] = complete_dataset['Country/Region'].apply(country_to_continent)
df_confirm_melt['Province/State'].fillna('All Province', inplace=True)
df_confirm_melt['Continent'] = df_confirm_melt['Country/Region'].apply(country_to_continent)

#  Add Month column for monthly aggregation 
complete_dataset['Month'] = complete_dataset['Date'].dt.to_period("M")

#  Create monthly dataframe: last record of each month per country 
monthly_df = complete_dataset.groupby(['Month', 'Country/Region']).tail(1)

#  Set index to Date for time-based slicing 
complete_dataset.set_index('Date', inplace=True)

#  Example: get data for a specific date and the top 10 countries by confirmed cases per continent 
x = complete_dataset.loc['2021-5-29']
top20_df = (
    x.groupby('Continent', group_keys=False)
      .apply(lambda x: x.nlargest(10, 'Confirm'))
)

#  Calculate recovery and death rates for each row 
complete_dataset['Recovery_rates'] = round((complete_dataset['Recover'] / complete_dataset['Confirm']) * 100, 2)
complete_dataset['Death_rates'] = round((complete_dataset['Death'] / complete_dataset['Confirm']) * 100, 2)

#  Create monthly confirmed dataframe for all countries 
monthly_confirm = complete_dataset.loc[pd.date_range('2020-1-30', '2021-5-29', freq=pd.Timedelta(days=30))]

# Group by date and country, sum Confirm, Recover, Death, and sort by Confirm descending 
gr_coun_date = monthly_confirm.groupby([monthly_confirm.index, 'Country/Region'])[['Confirm', 'Recover', 'Death']].sum()
gr_coun_date = gr_coun_date.reset_index().sort_values('Confirm', ascending=False)


#  Sidebar country selection 
all_countries = complete_dataset['Country/Region'].unique()
all_countries = np.insert(all_countries, 0, 'World')  # Add 'World' as the first option

st.sidebar.title('Customizer')
country = st.sidebar.selectbox('Choose the Country', options=all_countries)

#  Main title and label assignment for map highlighting 
st.title(country + ' (Covid19) Dashboard')
complete_dataset['label'] = complete_dataset['Country/Region'].apply(lambda x: x if x == country else '')

# Set index for monthly_df for easier time-based slicing 
monthly_df.set_index('Date', inplace=True)

#  Set default map center coordinates 
center_lat = 0
center_lon = 0
if country != 'World':
    # Center map on selected country
    selected_row = complete_dataset[complete_dataset['Country/Region'] == country].iloc[0]
    center_lat = selected_row['Lat']
    center_lon = selected_row['Long']

# --- Sidebar date selection ---
st.sidebar.write('Select a date')
from1 = st.sidebar.date_input('From', value=min_date)
to = st.sidebar.date_input('To', value=max_date)
from2 = pd.to_datetime(from1)
to1 = pd.to_datetime(to)

#  Format dates for display 
start = from1.strftime('%Y-%m-%d')
end = to.strftime('%Y-%m-%d')

#  Sidebar status selection 
Status = st.sidebar.selectbox('Choose Status', options=['Confirm', 'Recover', 'Death', 'Overall Analysis'])

#  Display current analysis selection at the top of the page 
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; width: 100%; font-size: 18px;">
        <div><strong> An AI powered Covid-19 Dashboard</strong></div>
        <div><strong>From:</strong> {start} &nbsp;&nbsp;&nbsp; <strong>To:</strong> {end}</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

#  Helper function to show death and recovery rates as metrics 
def RateMetric(from1, to, df_rate, country):
    if country != 'World':
        death_rate = round((df_rate.loc[to]['Death'] / df_rate.loc[to]['Confirm']) * 100, 2)
        recover_rate = round((df_rate.loc[to]['Recover'] / df_rate.loc[to]['Confirm']) * 100, 2)
        MetricDesign(death_rate, '', 'Death Rate (%)')
        st.markdown("<br>", unsafe_allow_html=True)
        MetricDesign(recover_rate, '', 'Recover Rate (%)')

    else:
        # Find the country with the highest death rate
        highest_death_rate = df_rate['Death_rate'].max()
        highest_death_rate_Country = df_rate[df_rate['Death_rate'] == highest_death_rate].index
        # Find the country with the lowest death rate
        lowest_death_rate = df_rate['Death_rate'].min()
        lowest_death_rate_Country = df_rate[df_rate['Death_rate'] == lowest_death_rate].index
        # Find the country with the highest recovery rate
        highest_recover_rate = df_rate['Recover_rate'].max()
        highest_recover_rate_Country = df_rate[df_rate['Recover_rate'] == highest_recover_rate].index

        # Display metrics for highest death and recovery rates
        MetricDesign(highest_death_rate, highest_death_rate_Country[0], 'Highest death rate(%) ')
        st.markdown("<br>", unsafe_allow_html=True)
        MetricDesign(highest_recover_rate, highest_recover_rate_Country[0], 'Highest recover rate(%) ')
        # MetricDesign(highest_confirm_cases,highest_confirm_cases_country,'highest confirm cases ')

def Donut(df_rate, country):
    # Prepare data for donut (pie) chart showing Confirmed, Recovered, and Death totals
    total_confirm = df_rate['Confirm'].sum()
    total_recover = df_rate['Recover'].sum()
    total_death = df_rate['Death'].sum()
    param = ['confirm', 'recover', 'death']
    value = [total_confirm, total_recover, total_death]
    sum_data = pd.DataFrame({
        'Category': param,
        'Values': value,
    })
    # Create and display the donut chart
    fig = px.pie(sum_data, names='Category', values='Values', hole=0.4)
    st.plotly_chart(fig)

def HighestCases(df_rate, country):
    # If not World, filter for the selected country only
    if country != 'World':
        df_rate = df_rate[df_rate['Country/Region'] == country]
    # Find the highest values for each metric
    highest_death_cases = df_rate['Death'].max()
    highest_recover_cases = df_rate['Recover'].max()
    highest_confirm_cases = df_rate['Confirm'].max()
    # Find the country (or index) with the highest value for each metric
    highest_death_cases_country = df_rate[df_rate['Death'] == highest_death_cases]['Country/Region'].values[0]
    highest_recover_cases_country = df_rate[df_rate['Recover'] == highest_recover_cases]['Country/Region'].values[0]
    highest_confirm_cases_country = df_rate[df_rate['Confirm'] == highest_confirm_cases]['Country/Region'].values[0]
    # Display metrics using custom design
    MetricDesign(int(highest_recover_cases), highest_recover_cases_country, 'Highest recover cases ')
    st.markdown("<br>", unsafe_allow_html=True)           
    MetricDesign(int(highest_death_cases), highest_death_cases_country, 'Highest death cases ')
    st.markdown("<br>", unsafe_allow_html=True)
    MetricDesign(int(highest_confirm_cases), highest_confirm_cases_country, 'Highest confirm cases ')

def MetricDesign(value_confirm,arrow, label):
    # Format and display a metric with a custom HTML card
    label = label
    arrow = arrow
    try:
        value = f"{(value_confirm):,}"
    except (ValueError, TypeError):
        value = "N/A"
    color = "green"
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 0.5rem; padding: 1rem; text-align: center">
            <div style="font-size: 1rem; color: gray;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
            <div style="font-size: 1rem; color: gray;">{arrow}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def geoScatter(df, status, country, color_scale, center_lon, center_lat):
    # Show a geographic scatter plot of COVID-19 data
    if status == 'Overall Analysis':
        status = 'Confirm'
    fig = px.scatter_geo(
        df,
        lat='Lat',
        lon='Long',
        color=status,       # Bubble color = confirmed cases
        color_continuous_scale=color_scale,
        hover_name='Country/Region',
        text='label',
        projection='natural earth',
        hover_data={'Country/Region': True, 'Lat': False, 'Long': False},
    )

    if country != 'World':
        # Highlight selected country and zoom in
        fig.update_traces(textposition='top center', textfont=dict(size=15, color='red', family='Arial'))
        fig.update_geos(
            showland=True, landcolor="lightgray",
            showcountries=True,
            showocean=True, oceancolor="lightblue",
            center=dict(lat=center_lat, lon=center_lon),  # Center the view
            lonaxis_range=[center_lon - 60, center_lon + 40],  # Horizontal zoom
            lataxis_range=[center_lat - 20, center_lat + 20],  # Vertical zoom
        )
    else:
        # For 'World' view, set default text styling and map aesthetics
        fig.update_traces(textposition='top center', textfont=dict(size=12, color='black'))

        # Improve map aesthetics for global view
        fig.update_geos(
            showland=True,
            landcolor="lightgray",
            showcountries=True,
            showocean=True,
            oceancolor="lightblue",
            showframe=True,
            framecolor="black"
        )

    # Set layout margins and display the map
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.header("COVID-19 Map For " + country)
    st.plotly_chart(fig, use_container_width=True)

def TotalCases(from1, to, df, status, country):
    """
    Display the total number of cases for the selected status and country.
    """
    if country == 'World':
        total = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        total = df.loc[to][status].sum()
    MetricDesign(int(total),'',  status)

def maxSpike(from1, to, df, status, country):
    """
    Display the maximum single-day spike for the selected status and country.
    """
    if country == 'World':
        val = df[status].max()
    else:
        df = df[df['Country/Region'] == country]
        val = df[status].max()
    status = 'Single Day Peak'
    MetricDesign(int(val),'',  status)

def OneDay(from1, to, df, status, country):
    """
    Display the value for the last 24 hours for the selected status and country.
    """
    if country == 'World':
        val = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        val = df.loc[to][status]
    status = 'Reported in Last 24 Hr'
    MetricDesign(int(val), '', status)
  
# Function to display the sum of cases for a given status and country on a specific date
def SDay(from1, to, df, status, country):
    if country == 'World':
        val = df.loc[to][status].sum()
    else:
        df = df[df['Country/Region'] == country]
        val = df[status].sum()
    MetricDesign(int(val),'',  '7 day moving average')

# Function to calculate and display active cases (Confirmed - (Deaths + Recovered))
def Active_cases(from1, to, df, status, country):
    if country == 'World':
        df = df.loc[to]
        val = df['Confirm'].sum() - (df['Death'].sum() + df['Recover'].sum())
    else:
        df = df[df['Country/Region'] == country]
        df = df.loc[to]
        val = df['Confirm'].sum() - (df['Death'].sum() + df['Recover'].sum())

    MetricDesign(int(val), '', status)

# Function to display COVID-19 metrics in a four-column layout using Streamlit
def MetricStruct(from2, to1, df_confirm_melt, status, country):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if status == 'Overall Analysis':
            r = 'Confirm'
            TotalCases(from2, to1, df_confirm_melt, r, country)
        else:
            TotalCases(from2, to1, df_confirm_melt, status, country)

    with col2:
        if status == 'Overall Analysis':
            a = 'Death'
            TotalCases(from2, to1, df_confirm_melt, a, country)
        else:
            x = 'Single_' + status
            maxSpike(from2, to1, df_confirm_melt, x, country)

    with col3:
        if status == 'Overall Analysis':
            j = 'Recover'
            TotalCases(from2, to1, df_confirm_melt, j, country)
        else:
            x = 'Single_' + status
            OneDay(from2, to1, df_confirm_melt, x, country)

    with col4:
        if status == 'Overall Analysis':
            k = 'Active'
            Active_cases(from2, to1, df_confirm_melt, k, country)
        else:
            y = '7Day_' + status
            SDay(from2, to1, df_confirm_melt, y, country)

# Function to plot a sunburst chart for continent and country breakdown
def SunPlot(df, status):
    col1, col2 = st.columns([3, 1])
    with col1:
        top20_df = (
            df.groupby('Continent', group_keys=False)
            .apply(lambda x: x.nlargest(10, Status))
        )
        plt.figure(figsize=(15, 15))
        fig = px.sunburst(
            df,
            path=['Continent', 'Country/Region'],
            values=status,
            color='Continent',
            color_discrete_map={
                'Asia': 'lightblue',
                'Europe': 'lightgreen'
            }
        )
        fig.update_layout(width=500, height=600)
        fig.update_traces(textfont_size=18)
        st.plotly_chart(fig)

# Function to calculate and display the rate (death or recovery) for a country or world
def Rate(from1, to, df, country, status):
    if country == 'World':
        df = df.loc[to]
        val = round((df[status].sum() / df['Confirm'].sum()) * 100, 2)
    else:
        df = df[df['Country/Region'] == country]
        df = df.loc[to]
        val = (df[status].sum() / df['Confirm'].sum()) * 100
    MetricDesign((val), '', status + '_Rate of the ' + country+'(%)')

#  Create two main tabs: Data Dashboard and Data AI 
tab1, tab2 = st.tabs(['Data Dashboard', 'Data AI'])

with tab1:
    #  If user selects 'Overall Analysis' status 
    if Status == 'Overall Analysis':        
        if country == 'World':
            #  Filter data for selected date range 
            df = complete_dataset
            mask = (complete_dataset.index >= from2) & (complete_dataset.index <= to1)
            df1 = complete_dataset[mask]
            #  Show main metrics for the world 
            MetricStruct(from2, to1, df1, Status, country)
            #  Get data for the last selected date 
            df2 = df1.loc[to1]
            #  Get top 10 countries by confirmed cases 
            top_10 = df2.sort_values('Confirm', ascending=False).head(10)['Country/Region'].unique()
            #  Filter data for top 10 countries 
            line_df = df[df['Country/Region'].isin(top_10)]
            grouped = df1.loc[to1]
            #  Prepare monthly bar chart data for top 10 countries 
            bar_df = line_df.loc[pd.date_range(from2, to1, freq='31D')].sort_values('Confirm', ascending=False)
            
            # Define color scale for map visualisation 
            color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
            ]
            #  Show map and rates side by side 
            col1, col2 = st.columns([3, 1])
            with col1:
                with st.spinner("Generating map..."):
                    geoScatter(grouped, Status, country, color_scale, center_lon, center_lat)
            with col2:
                st.subheader('')
                #  Show death and recovery rates 
                Rate(from2, to1, df, country, 'Death')
                st.markdown("<br>", unsafe_allow_html=True)
                Rate(from2, to1, df, country, 'Recover')
                st.markdown("<br>", unsafe_allow_html=True)
                #  Calculate rates for each country 
                df_rate = df.groupby('Country/Region')[['Confirm', 'Recover', 'Death']].sum()
                df_rate['Death_rate'] = round((df_rate['Death'] / df_rate['Confirm']) * 100, 2)
                df_rate['Recover_rate'] = round((df_rate['Recover'] / df_rate['Confirm']) * 100, 2)
                highest_death_rate = df_rate['Death_rate'].max()
                #  Show metrics for highest/lowest rates 
                RateMetric(from2, to1, df_rate, country) 
                st.markdown("<br>", unsafe_allow_html=True)           
                
            #  Donut chart and highest cases metrics 
            col1, col2 = st.columns([3, 1])  
            with col1:
                st.subheader('Case Status Overview')
                Donut(df_rate, country)
            with col2:
                HighestCases(df1, country)
                
            #  Bar chart for top 10 countries (Confirm, Recover, Death) 
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader('Top 10 Countries Total_Confirm vs Total_Recovery vs Total_Death')
                fig = px.bar(
                    grouped[grouped['Country/Region'].isin(top_10)][['Country/Region', 'Confirm', 'Death', 'Recover']],
                    x='Country/Region',
                    y=['Confirm', 'Recover', 'Death'],
                    barmode='group'
                )
                st.plotly_chart(fig)
            #  Area plot for monthly recovery rates of top 10 countries 
            data = df1[df1['Country/Region'].isin(top_10)][['Country/Region', 'Recovery_rates']]
            data_new = data.loc[pd.date_range(from2, to1, freq='35D')].reset_index().rename(columns={'index':'Date'}).set_index('Date')
            col1, col2 = st.columns([1, 3])
            with col2:
                st.subheader('Top 10 Countries Monthly Recovery Cases Area Plot')
                fig = px.area(
                    data_new,
                    x=data_new.index,
                    y='Recovery_rates',
                    color='Country/Region'
                )
                st.plotly_chart(fig)
            with col1:
                #  Show metrics for the highest recovery and death rates among countries with high cases 
                data = df1[df1['Confirm'] > 3000000]
                data1 = df1[df1['Death'] > 100000]
                gr = data.groupby('Country/Region')['Recovery_rates'].mean().sort_values(ascending=False).head(1)
                gr1 = data1.groupby('Country/Region')['Death_rates'].mean().sort_values(ascending=False).head(1)
                highest_recovery_rate = gr.values[0]
                highest_recovery_country =  gr.index[0]
                highest_death_rate =  gr1.values[0]
                highest_death_country = gr1.index[0]
                
                MetricDesign(round(highest_recovery_rate,2),highest_recovery_country,  'Highest Recovery Rate (%)')
                st.markdown("<br>", unsafe_allow_html=True)
                MetricDesign( round(highest_death_rate,2),highest_death_country, ' Highest Death Rate (%)')            

        else:   
            #  Country-specific analysis for 'Overall Analysis' 
            df = complete_dataset
            mask = (complete_dataset.index >= from2) & (complete_dataset.index <= to1)
            df1 = complete_dataset[mask]
            mask1 = (monthly_df.index >= from2) & (monthly_df.index <= to1)
            grouped = df1.loc[to1]
            df1 = df1[df1['Country/Region'] == country]
            #  Monthly data for the selected country 
            df1_month = df1.loc[pd.date_range(from2, to1, freq=pd.Timedelta(days=35))].sort_index(ascending=False)
            monthly_df1 = monthly_df[mask1]
            monthly_df1 = monthly_df1[monthly_df1['Country/Region'] == country]
            #  Province/state-level data for the last selected date 
            new_df = df_confirm_melt.loc[to1]
            df_new = new_df[new_df['Country/Region'] == country]
            no_of_province = len(df_new)
            
            #  Show main metrics for the country 
            MetricStruct(from2, to1, df, Status, country)
            col1, col2 = st.columns([3, 1])
            color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
            ]
            with col1:
                with st.spinner("Generating map..."):
                    geoScatter(grouped, Status, country, color_scale, center_lon, center_lat)
            with col2:
                st.header('')
                RateMetric(from2, to1, df1, country)
            #  Pie chart for province/state breakdown if only one province 
            col1,col2 = st.columns([3,1])
            with col1:
                Confirm = df1.loc[to1]['Confirm']
                Recover = df1.loc[to1]['Recover']
                Death = df1.loc[to1]['Death']
                df_donut = pd.DataFrame({
                    'Category': ['Confirm', 'Recover', 'Death'],
                    'Values': [Confirm, Recover, Death],
                })
                fig = px.pie(df_donut, names='Category', values='Values', hole=0.5)
                st.header('Death vs Recover vs Confirm  ')
                st.plotly_chart(fig)
            #  Line chart for Confirm, Recover, Death over time 
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.line(
                    monthly_df1,
                    x=monthly_df1.index,
                    y=['Confirm', 'Recover', 'Death'],                
                )
                st.header('Daily Line Chart for ' + country)
                st.plotly_chart(fig)
            #  Bar chart for Confirm, Recover, Death over time 
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.bar(
                    monthly_df1,
                    x=monthly_df1.index,
                    y=['Confirm', 'Recover', 'Death'],
                )
                st.header('Monthly Bar Chart for ' + country)
                st.plotly_chart(fig)
            #  Line chart for daily new cases (Single_Confirm, Single_Recover, Single_Death) 
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.line(
                    df1,
                    x=df1.index,
                    y=['Single_Confirm', 'Single_Recover', 'Single_Death'],
                )
                st.header('Monthly Bar Chart for ' + country)
                st.plotly_chart(fig)
            with col2:
                SDay(from1, to, df1, '7Day_Confirm', country)

    else:
        if country == 'World':
            df = complete_dataset
            mask = (complete_dataset.index >= from2) & (complete_dataset.index <= to1)
            df1 = complete_dataset[mask]
            MetricStruct(from2, to1, df1, Status, country)
            df2 = df1.loc[to1]
            # Get top 10 countries by selected status
            top_10 = df2.sort_values('Confirm', ascending=False).head(10)['Country/Region'].unique()
            line_df = df[df['Country/Region'].isin(top_10)]
            grouped = df1.loc[to1]
            # Prepare bar chart data for top 10 countries (monthly)
            bar_df = line_df.loc[pd.date_range(from2, to1, freq='31D')].sort_values('Confirm', ascending=False).reset_index().rename(columns={'index':'Date'}).set_index('Date')
            # Get top 30 countries for extended line chart
            top_20 = df2.sort_values('Confirm', ascending=False).head(30)['Country/Region'].unique()
            line_df_30 = df[df['Country/Region'].isin(top_20)]

            # Define color scale for map
            color_scale = [
                [0.0, "green"],
                [0.25, "yellow"],
                [0.5, "orange"],
                [1.0, "red"]
            ]
            # Show world map with bubbles for selected status
            geoScatter(grouped, Status, country, color_scale, center_lon, center_lat)

            # Continent-wise analysis and sunburst chart
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader('Continent Wise ' + Status + ' cases Analysis')
                SunPlot(grouped, Status)
            with col2:
                st.header('')
                # Find the continent with the maximum cases for the selected status
                g = grouped.groupby('Continent')[Status].sum().sort_values(ascending=False).head(1)
                h = grouped.groupby('Continent')[Status].sum().sort_values().head(1)
                continent_case = g.values[0]
                continent_name = g.index[0]
                MetricDesign(continent_case,continent_name , 'Maximum ' + Status)
                continent_case1 = h.values[0]
                continent_name2 = h.index[0]
                st.markdown("<br>", unsafe_allow_html=True)
                MetricDesign(continent_case1,continent_name2,  'Minimum ' + Status)
            # Plot over time cases
            df_val = df1.groupby(df1.index)[['Confirm', 'Death', 'Recover']].sum()
            st.subheader( Status + ' cases over the time')
            fig = px.bar(
                df_val,
                x = df_val.index,
                y = Status

            )
            st.plotly_chart(fig)
            # Line chart for the top 10 countries
            st.subheader('Top 10 countries daily ' + Status + ' cases Line Chart')
            fig = px.line(
                line_df,
                x=line_df.index,
                y=Status,
                color='Country/Region',
            )
            st.plotly_chart(fig)

            # Bar chart for top 10 countries (monthly)
            st.subheader('Top 10 countries daily ' + Status + ' cases Bar Chart')
            fig = px.bar(
                bar_df,
                x=bar_df.index,
                y=Status,
                color='Country/Region',
            )
            st.plotly_chart(fig)        

        else:   
            # Filter the main DataFrame for the selected date range and country
            df = complete_dataset
            mask = (complete_dataset.index >= from2) & (complete_dataset.index <= to1)
            df1 = complete_dataset[mask]
            mask1 = (df_confirm_melt.index >= from2) & (df_confirm_melt.index <= to1)
            mask2 = (df_recover_melt.index >= from2) & (df_recover_melt.index <= to1)
            mask3 = (df_death_melt.index >= from2) & (df_death_melt.index <= to1)
            grouped = df1.loc[to1]
            df1 = df1[df1['Country/Region'] == country]
            # Create a monthly DataFrame for the selected country
            df1_month = df1.loc[pd.date_range(from2,to1,freq = pd.Timedelta(days = 35))].sort_index(ascending = False).reset_index().rename(columns={'index':'Date'}).set_index('Date')
            
            # Prepare province/state-level data for the selected status
            if Status =='Confirm':
                new_df = df_confirm_melt[mask1]
                new_df = new_df[new_df['Country/Region'] == country]
                ld = new_df.loc[to1]
                # Calculate daily new confirmed cases per province/state
                new_df['Single_Confirm']=new_df.groupby('Province/State')['Confirm'].diff().fillna(0)
            else:
                if Status == 'Recover':
                    new_df = df_recover_melt[mask2]
                    new_df = new_df[new_df['Country/Region'] == country]
                    ld = new_df.loc[to1]
                    # Calculate daily new recoveries per province/state
                    new_df['Single_Recover']=new_df.groupby('Province/State')[Status].diff().fillna(0)
                else:
                    new_df = df_death_melt[mask3]
                    new_df = new_df[new_df['Country/Region'] == country]
                    ld = new_df.loc[to1]
                new_df['Single_Death']=new_df.groupby('Province/State')[Status].diff().fillna(0)
            df_new = new_df[new_df['Country/Region']==country]
            # Count the number of provinces/states for the country
            no_of_province = len(df_new['Province/State'].unique())
            
            # Show main metrics for the selected country and status
            MetricStruct(from2,to1,df1,Status,country)
            col1,col2 = st.columns([3,1])
            color_scale = [
            [0.0, "green"],
            [0.25, "yellow"],
            [0.5, "orange"],
            [1.0, "red"]
            ]
            
            with col1:
                # Show geo scatter map for the selected country and status
                geoScatter(grouped,Status,country,color_scale,center_lon,center_lat)
            with col2:
                st.header('DataFrame')
                # Display DataFrame with confirmed and daily new confirmed cases
                st.write(df1[['Confirm','Single_Confirm']])
            col1,col2 = st.columns([3,1])
            with col1:
                # Plot daily new cases and 7-day moving average for the selected status
                y1 = '7Day_'+Status
                y2 = 'Single_'+ Status
                fig = px.line(df1,
                x = df1.index,
                y = [y2,y1],
                color_discrete_sequence=['red', 'yellow']
                )
                st.header('Daily New Cases vs 7Days Moving Average in '+ country)
                st.plotly_chart(fig)
            if no_of_province>1:
                col1,col2 = st.columns([3,1])
                with col1:
                    # Show province/state-wise distribution as a pie chart
                    fig = px.pie(df_new,names = 'Province/State',values = Status,hole = 0.5)
                    st.header('Province wise cases Percentage')
                    st.plotly_chart(fig)
                with col2:
                # Show metrics for province/state with maximum cases and single-day spike
                    if Status == 'Confirm':
                        max_province_df = ld[ld[Status] == ld[Status].max()]
                        max_province_name = max_province_df['Province/State'].values[0]
                        num = max_province_df[Status].values[0]
                        
                        Single_day_confirm = new_df[new_df['Single_Confirm']==new_df['Single_Confirm'].max()]['Single_Confirm'].values[0]
                        Single_day_confirm_country = new_df[new_df['Single_Confirm']==new_df['Single_Confirm'].max()]['Province/State'].values[0]
                        
                        MetricDesign(num,max_province_name,'Maximum '+ Status)
                        MetricDesign(Single_day_confirm,Single_day_confirm_country,'Single_day Maximum Spike ')
                    if Status == 'Recover':
                        max_province_df = ld[ld[Status] == ld[Status].max()]
                        max_province_name = max_province_df['Province/State'].values[0]
                        num = max_province_df[Status].values[0]
                        
                        Single_day_confirm = new_df[new_df['Single_Recover']==new_df['Single_Recover'].max()]['Single_Recover'].values[0]
                        Single_day_recover_country = new_df[new_df['Single_Recover']==new_df['Single_Recover'].max()]['Province/State'].values[0]
                        
                        MetricDesign(num,max_province_name,'Maximum '+ Status)
                        MetricDesign(Single_day_confirm,Single_day_recover_country,'Single_day Maximum Spike '+ Status)
                    if Status == 'Death':
                        max_province_df = ld[ld[Status] == ld[Status].max()]
                        max_province_name = max_province_df['Province/State'].values[0]
                        num = max_province_df[Status].values[0]
                        Single_day_death = new_df[new_df['Single_Death']==new_df['Single_Death'].max()]['Single_Death'].values[0]
                        Single_day_death_country = new_df[new_df['Single_Death']==new_df['Single_Death'].max()]['Province/State'].values[0]
                        MetricDesign(num,max_province_name,'Maximum '+ Status)
                        MetricDesign(Single_day_death,Single_day_death_country,'Single_day Maximum Spike '+ Status)

                # Daily line chart for the selected country and status
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.line(
                    df1,
                    x=df1.index,
                    y=Status,
                )
                st.header(Status +'cases over the time for ' + country)
                st.plotly_chart(fig)

            # Moonthly bar chart for the selected country and status
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.bar(
                    df1_month,
                    x=df1_month.index,
                    y=Status,
                )
                st.header('Monthly cumulative '+Status +' cases for ' + country)
                st.plotly_chart(fig)

# Filter the main DataFrame for the selected date range
mask = (complete_dataset.index >= from2) & (complete_dataset.index <= to1)
df_selected = complete_dataset[mask]
if country != "World":
    df_selected = df_selected[df_selected['Country/Region'] == country]

# Get the last day in the selected range for summary stats
try:
    last_day = df_selected.loc[to1]
except:
    last_day = df_selected.iloc[-1]

# Calculate total confirmed, recovered, and death cases
total_confirm = int(last_day['Confirm'].sum()) if country == "World" else int(last_day['Confirm'])
total_recover = int(last_day['Recover'].sum()) if country == "World" else int(last_day['Recover'])
total_death = int(last_day['Death'].sum()) if country == "World" else int(last_day['Death'])

# Calculate death and recovery rates as percentages
death_rate = round((total_death / total_confirm) * 100, 2) if total_confirm > 0 else 0
recovery_rate = round((total_recover / total_confirm) * 100, 2) if total_confirm > 0 else 0

# Calculate active cases (confirmed minus recovered and deaths)
active_cases = total_confirm - (total_recover + total_death)

# Find the highest single-day spike in new confirmed cases
if country == "World":
    max_spike = int(df_selected['Single_Confirm'].sum())
else:
    max_spike = int(df_selected['Single_Confirm'].max())

# Calculate the last 24-hour change in confirmed cases
if len(df_selected) > 1:
    last_24hr = int(df_selected['Single_Confirm'].iloc[-1])
else:
    last_24hr = 0

# Get the top provinces/regions by confirmed cases (if available)
if country != "World" and 'Province/State' in df_confirm_melt.columns:
    mask_prov = (df_confirm_melt.index >= from2) & (df_confirm_melt.index <= to1)
    prov_df = df_confirm_melt[mask_prov]
    prov_df = prov_df[prov_df['Country/Region'] == country]
    top_provinces = prov_df.groupby('Province/State')['Confirm'].sum().sort_values(ascending=False).head(3).index.tolist()
    top_provinces = ', '.join(top_provinces)
else:
    top_provinces = "N/A"



prompt = f"""
You are an expert data analyst. Please help the user understand the COVID-19 situation based on their selection. 
Break down each metric in simple language, and present the information in a clear, structured, and friendly way.
Use headings, bullet points, and short paragraphs. Explain what each number means and why it matters.

User Selection:
- **Country:** {country}
- **Date Range:** {start} to {end}
- **Status:** {Status}

Please provide the following breakdown:

1. **Overview**
   - Briefly introduce the selected country and date range.
   - Mention the status type selected.

2. **Key Numbers**
   - **Total Confirmed Cases:** {total_confirm}
     - Explain what this means.
   - **Total Recoveries:** {total_recover}
     - Explain what this means.
   - **Total Deaths:** {total_death}
     - Explain what this means.
   - **Active Cases:** {active_cases}
     - Explain what this means.

3. **Rates**
   - **Death Rate:** {death_rate}%
     - What does this percentage mean for the country?
   - **Recovery Rate:** {recovery_rate}%
     - What does this percentage mean for the country?

4. **Trends and Spikes**
   - **Highest Single-Day Spike:** {max_spike}
     - On which day did this happen, if possible?
     - Why is this important?
   - **Last 24-hour Change:** {last_24hr}
     - What does this tell us about the current trend?

5. **Regional Breakdown**
   - **Top Provinces/Regions:** {top_provinces}
     - List the top regions and their significance.

6. **Insights**
   - Highlight any notable trends, spikes, or anomalies.
   - Offer a simple interpretation of whether the situation is improving, stable, or worsening.

7. **Summary**
   - End with a friendly, encouraging note or a call to action (e.g., "Stay safe!" or "Keep following health guidelines.").

Please use simple language and make the summary easy for anyone to understand, even without a data background.
"""



with tab2:
    st.markdown("## 🤖 Data AI Assistant")
    tab_summary, tab_ask = st.tabs(["Data Overview by AI", "Ask to AI"])

    with tab_summary:
        if st.button("🧠 Generate AI Overview"):
            with st.spinner("AI is thinking..."):
                try:
                    response = model.generate_content(
                        f"Write a detailed, interactive, and easy-to-understand summary from this COVID-19 data:\n{prompt}"
                    )
                    st.success("Here is your Overview:")
                    st.markdown(f"📢 **AI Summary:**\n\n{response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_ask:
        user_question = st.text_area("Ask anything about this data:", key="user_question_box")
        if st.button("Ask AI"):
            if user_question.strip():
                with st.spinner("AI is thinking..."):
                    try:
                        reply_prompt = f"""You are a helpful data assistant. The user is viewing this COVID-19 data:{prompt}.
                        Now answer the following question specifically, using only the information above. If the answer is not in the data, say so.
                        User question: {user_question}"""
                        qa_response = model.generate_content(reply_prompt)
                        st.markdown(f"🤖 **AI Answer:**\n\n{qa_response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please enter a question to ask the AI.")
