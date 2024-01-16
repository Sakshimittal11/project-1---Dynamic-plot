#!/usr/bin/env python
# coding: utf-8

# # SETTING UP DATA  

# In[1]:


from datetime import date

current_date = date.today()
print("Today's date:", current_date)


# In[2]:


pip install watermark


# In[165]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser
import watermark

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[166]:


import plotly.express as px
import plotly.graph_objects as go


# In[167]:


from pathlib import Path


# In[168]:


df_confirmed =pd.read_csv('time_series_covid19_confirmed_global.csv')
df_confirmed.head()


# In[169]:


df_confirmed.shape


# In[170]:


df_deaths =pd.read_csv('time_series_covid19_deaths_global.csv')
df_deaths.head()


# In[171]:


df_deaths.shape


# In[172]:


def restructure(dfname, idvar):
    df_use = dfname.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date', value_name=idvar)
    df_use.drop(['Lat', 'Long'], axis=1, inplace=True)
    df_use.sort_values(by=['Country/Region','Province/State', 'date'], inplace=True)
    
    return df_use


# In[173]:


df_confirmed = restructure(df_confirmed, idvar='cases')
df_confirmed.shape


# In[174]:


df_confirmed.head(3)


# In[175]:


df_deaths = restructure(df_deaths, idvar='deaths')
df_deaths.shape


# In[176]:


df_deaths.head()


# In[177]:


results = pd.merge(df_confirmed, df_deaths, on=['Country/Region','Province/State', 'date'])


# In[178]:


results[["date2"]] = results[["date"]].apply(pd.to_datetime)


# In[179]:


#results.dtypes


# In[180]:


results.shape


# In[181]:


results.head()


# In[182]:


pd.set_option('display.max_rows', 200)


# In[183]:


print(results[results['Country/Region'] == "Turkey"][:15])


# In[184]:


results['Country/Region'].mask(results['Province/State'] == 'Hong Kong', 'Hong Kong China', inplace=True)
# rename "South_Korea"
results['Country/Region'].mask(results['Country/Region'] == 'Korea, South', 'South Korea', inplace=True)
# name "Hubei" province in China where Wuhan is located
results['Country/Region'].mask(results['Province/State'] == 'Hubei', 'Hubei China', inplace=True)


# In[185]:


print(results[results['Country/Region'] == "Hong Kong China"][:5])
print(results[results['Country/Region'] == "South Korea"][:5])
print(results[results['Country/Region'] == "Hubei China"][:5])


# In[186]:


country_collapse = results.groupby(['Country/Region','date','date2']).sum()


# In[187]:


type(country_collapse)


# In[188]:


country_collapse.reset_index(inplace = True)


# In[189]:


print(country_collapse.head(5))


# In[190]:


#print(country_collapse[country_collapse['Country/Region'] == "France"][:])
#print(country_collapse[country_collapse['Country/Region'] == "Australia"][:])


# In[191]:


df_cases_deaths = country_collapse.copy()


# In[192]:


df_cases_deaths.head(3)


# In[193]:


df_cases_deaths= df_cases_deaths.drop(["date"], axis=1)
df_cases_deaths
df_cases_deaths.rename({'date2': 'date'}, axis=1, inplace=True)
df_cases_deaths['date_n'] = df_cases_deaths['date']
df_cases_deaths['date'] = df_cases_deaths['date'].astype(str)


# In[194]:


df_cases_deaths.dtypes


# In[195]:


df_cases_deaths.head(3)


# In[196]:


#print(df_cases_deaths[df_cases_deaths['Country/Region'] == "Vietnam"][:])


# In[197]:


dfall = df_cases_deaths.copy()


# In[198]:


# it is working
def get_date_first_case(dfin):
    dfin.sort_values(by=['Country/Region', 'date'], inplace=True)

    dfuse = dfin.copy()
    dfuse['dummy'] = 1
    #print(dfuse.head())
    first = dfuse.groupby(['Country/Region', 'cases', 'dummy']).first()
    print(first.head(3))
    first.reset_index(inplace = True)
    
    df_first_case = first[first['cases'] > 0]
    df_first_case2 = df_first_case.groupby(['Country/Region', 'dummy']).first()
    df_first_case2.reset_index(inplace = True)

    df_first_case2.rename({'date': 'date_first_case'}, axis=1, inplace=True)
    
    # keep only the two columsn we need:  country, date of first {case or death}
    df_first_case2 = df_first_case2[['Country/Region','date_first_case']]
    #print(df_first_case2.head())
    df_first_case2.sort_values(by=['Country/Region'])

    df_cases_deaths2 = pd.merge(dfin, df_first_case2, on=['Country/Region'])
    print("\n")
    #print(df_cases_deaths2.head())

    #print(df_cases_deaths2.dtypes)
    # get day difference
    
    # 8/10/21: updated code here
    df_cases_deaths2[['date_first_case2']] = df_cases_deaths2[['date_first_case']].apply(pd.to_datetime)
    #df_cases_deaths2[['date_first_caseb']] = df_cases_deaths2['date_first_case'].apply(pd.to_datetime)
    #print(df_cases_deaths2.dtypes)

    df_cases_deaths2['days_since_first_case'] = ((df_cases_deaths2['date_n'] - df_cases_deaths2['date_first_case2']).dt.days) + 1
    print(df_cases_deaths2.head())

    
    return df_cases_deaths2   


# In[199]:


df_day_first_case= get_date_first_case(dfall)


# In[200]:


# df with: date of first death


# In[201]:


# it is working
def get_date_first_death(dfin):
    dfin.sort_values(by=['Country/Region', 'date'], inplace=True)

    dfuse = dfin.copy()
    dfuse['dummy'] = 1
    #print(dfuse.head())
    first = dfuse.groupby(['Country/Region', 'deaths', 'dummy']).first()
    print(first.head(3))
    first.reset_index(inplace = True)
    
    df_first_case = first[first['deaths'] > 0]
    df_first_case2 = df_first_case.groupby(['Country/Region', 'dummy']).first()
    df_first_case2.reset_index(inplace = True)

    df_first_case2.rename({'date': 'date_first_death'}, axis=1, inplace=True)
    
    # keep only the two columsn we need:  country, date of first {case or death}
    df_first_case2 = df_first_case2[['Country/Region','date_first_death']]
    #print(df_first_case2.head())
    df_first_case2.sort_values(by=['Country/Region'])

    df_cases_deaths2 = pd.merge(dfin, df_first_case2, on=['Country/Region'], how="outer")
    print("\n")
    print(df_cases_deaths2.head())

    print(df_cases_deaths2.dtypes)
    # get day difference
    df_cases_deaths2[['date_first_death']] = df_cases_deaths2[['date_first_death']].apply(pd.to_datetime)
    df_cases_deaths2['days_since_first_death'] = ((df_cases_deaths2['date_n'] - df_cases_deaths2['date_first_death']).dt.days)+1
    #print(df_cases_deaths2.head())

    
    return df_cases_deaths2   


# In[202]:


df_day_first_cd= get_date_first_death(df_day_first_case)


# In[203]:


df_day_first_cd.head()


# In[204]:


print(df_day_first_case[df_day_first_case['Country/Region'] == "Vietnam"][:])


# In[205]:


# Don't delete this. 
df_with_day0 = df_day_first_cd.copy()


# In[206]:


usedf = df_with_day0.copy()
usedf.sort_values(["Country/Region", 'date'])
df = usedf.set_index(["Country/Region"])

df['cases_lag'] = df.groupby(['Country/Region'])['cases'].shift(1)
df.reset_index()

#print(df.head())
df['deaths_lag'] = df.groupby(['Country/Region'])['deaths'].shift(1)
df.reset_index()


df['daily_case_count'] = ((df['cases'] - df['cases_lag']))
df['daily_death_count'] = ((df['deaths'] - df['deaths_lag']))


#df.reset_index()
df = df.drop(["cases_lag", "deaths_lag"], axis=1)

df['daily_case_count'] = df['daily_case_count'].fillna(0).astype(np.int64)
df['daily_death_count']= df['daily_death_count'].fillna(0).astype(np.int64)
df.reset_index(inplace = True)

df.info()
df['days_c1_to_d1'] = (df['date_first_death'] - df['date_first_case2']).dt.days

df['date_first_case'] = df['date_first_case'].astype(str)
df['date_first_death'] = df['date_first_death'].astype(str)
df['day_of_case'] = df['days_since_first_case']
df['day_of_death'] = df['days_since_first_death']


# In[207]:


df.head(5)


# In[208]:


# check that lag is computed correctly
#print(df[df['Country/Region'] == "Hong Kong China"])
#test_check = (df[df['Country/Region'] == "Afghanistan"])
#test_check[25:30]


# In[209]:


dfall = df.copy()


# In[210]:


dfall.dtypes


# In[211]:


dfall.head()


# In[212]:


# check that date from first case/death gives correct number of days
#print(dfall[dfall['Country/Region'] == "Norway"])
#print(dfall[dfall['Country/Region'] == "Sweden"])


# In[213]:


dfall['day_name'] = dfall['date_n'].dt.day_name()
#dfall['day_name']


# In[214]:


print(dfall[dfall['Country/Region'] == "Vietnam"][:])


# In[215]:


dfall.info()


# In[216]:


one_row_per_country = dfall.groupby(['Country/Region', 'date_first_case', 'date_first_death', 'days_c1_to_d1']).last()
one_row_per_country.reset_index(inplace = True)

one_row_per_country = one_row_per_country[['Country/Region', 'date_first_case', 'date_first_death', 'days_c1_to_d1', 'date', 'cases', 'deaths' ]]
one_row_per_country


one_row_per_country.to_csv("time_series_covid19_confirmed_global.csv"+"country_level_data.csv")

#df.groupby('Country/Region').nunique().plot(kind='bar')

#plt.show()


# In[217]:


one_row_per_country['days_c1_to_d1'].describe()


# In[218]:


# plt.figure (figsize=(15,10))

# one_row_per_country['days_c1_to_d1'].plot.hist(grid=True, bins=15, rwidth=0.9,
#                    color='#607c8e')
# plt.title('Days from case 1 to death 1')
# plt.xlabel('Days')
# plt.ylabel('')
# plt.grid(axis='y', alpha=1.0)


# In[219]:


import plotly.offline as pyo
import plotly.graph_objs as go


# In[220]:


df = one_row_per_country.copy()

df.sort_values(by=['days_c1_to_d1'], inplace=True)
df.head(5)


# In[221]:


days_count= df.groupby(['days_c1_to_d1']).count()


# In[222]:


days_count.reset_index(inplace = True)
days_count.head(5)


# # BAR GRAPH

# In[224]:


data1 = [go.Bar(x = df['days_c1_to_d1'],
              y = df['Country/Region']),]


data2 = [go.Bar(x = days_count['days_c1_to_d1'],
              y = days_count['Country/Region']),]

layout = go.Layout(title='Days from First Case to First Death (frequency count)')

fig = go.Figure(data=data1, layout=layout)
#pyo.plot(fig)
fig.show()

# fig = go.Figure(data2, layout)
# fig.show()


# In[225]:


#imputing missing data here

newdf=dfall.copy()

maskv2 = (newdf['Country/Region'] == 'India') & (newdf['days_since_first_case'] == 337)

newdf['daily_case_count'].mask((maskv2), 20929, inplace=True)


maskv2 = (newdf['Country/Region'] == 'India') & (newdf['days_since_first_case'] == 345)

newdf['daily_case_count'].mask((maskv2), 18434, inplace=True)

maskv3 = (newdf['Country/Region'] == 'India') & (newdf['days_since_first_case'] == 346)

newdf['daily_case_count'].mask((maskv3), 18433, inplace=True)


# In[226]:


dfall = newdf.copy()


# In[227]:


dfall.head(5)


# In[228]:


#!conda install -c conda-forge pyarrow


# In[229]:


pip install feather


# In[230]:


import feather


# In[231]:


print("Today's date:", current_date)


# In[232]:


# NOTE: directly from documentation.  No legend

import numpy as np
import pandas as pd
import plotly.graph_objects as go  #plotly 4.0.0rc1

df = pd.read_csv('time_series_covid19_confirmed_global.csvcountry_level_data.csv')

df.head(3)


# In[236]:


import plotly.graph_objects as go 


# # LINE CHARTS

# In[240]:


#!pip install ipywidgets>=7.0.0


# In[241]:


#!pip install --upgrade pip


# In[242]:


# reference:  https://community.plotly.com/t/cumulative-lines-animation-in-python/25707/2
import numpy as np
import pandas as pd
import feather
from datetime import date

import plotly.graph_objects as go  


# In[243]:


today = date.today()
print(today)


# In[244]:


pip install feather-format


# In[245]:


import feather

dfall = df.copy()


# In[247]:


from datetime import date

today = date.today
print("Today's date:", today)


# In[248]:


import pandas as pd
import watermark
import feather
import plotly.express as px
get_ipython().run_line_magic('reload_ext', 'watermark')


# In[252]:


trace1 = go.Scatter(x=df[timeval][:2],
                    y=low[:2],
                    mode='lines',
                    line=dict(width=1.5),
                   name="aaaaa")

trace2 = go.Scatter(x = df[timeval][:2],
                    y = high[:2],
                    mode='lines',
                    line=dict(width=1.5),
                   name="bbbbb")

increment = 50
frames = [dict(data= [dict(type='scatter',
                           x=df[timeval][:k+increment],
                           y=low[:k+increment]),
                      dict(type='scatter',
                           x=df[timeval][:k+increment],
                           y=high[:k+increment])],
               traces= [0, 1],  # frames[k]['data'][0]  updates trace1, and   frames[k]['data'][1], trace2 
              )
          for k  in  range(1, len(low)-1)] 

layout = go.Layout(width=650,
                   height=400,
                   showlegend=True,
                   hovermode='closest',
                   updatemenus=[dict(type='buttons', showactive=False,
                                y=1.05,
                                x=1.15,
                                xanchor='right',
                                yanchor='bottom',
                                pad=dict(t=0, r=10),
                                buttons=[dict(label='Play',
                                              method='animate',
                                              args=[None, 
                                                    dict(frame=dict(duration=0.5, 
                                                                    redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True,
                                                         mode='immediate')])])])


layout.update(xaxis =dict(range=[df[timeval][0], df[timeval][len(df)-1]], autorange=False),
              yaxis =dict(range=[min_low-5.5, max_high+5.5], autorange=False));

fig = go.Figure(data=[trace1, trace2], frames=frames, layout=layout)

fig.show()


# # RUNNING BAR CHART OF CASES

# In[253]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Assuming df is your DataFrame
# If not, you can read your data into a DataFrame using pd.read_csv or pd.read_excel

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Running Bar Chart Over Time'])

# Initialize an empty bar chart
bar_chart = go.Bar(x=[], y=[])

# Add the bar chart to the subplot
fig.add_trace(bar_chart)

# Update layout
fig.update_layout(
    title='Running Bar Chart Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Cases',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Bar(x=df_transposed.index,
                                y=df_transposed.iloc[:, :k+1].iloc[:, -1])],
                   name=f'frame{k+1}') for k in range(1, len(df_transposed.columns))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# # ANIMATED HEATMAP OF CASES

# In[254]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Assuming df is your DataFrame
# If not, you can read your data into a DataFrame using pd.read_csv or pd.read_excel

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Animated Heatmap Over Time'])

# Initialize an empty heatmap
heatmap = go.Heatmap(z=[], x=[], y=[])

# Add the heatmap to the subplot
fig.add_trace(heatmap)

# Update layout
fig.update_layout(
    title='Animated Heatmap Over Time',
    xaxis_title='Country/Region',
    yaxis_title='Date',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Heatmap(z=df_transposed.iloc[:, :k+1].values,
                                    x=df_transposed.index,
                                    y=df_transposed.columns)],
                   name=f'frame{k+1}') for k in range(1, len(df_transposed.columns))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# # ANIMATED LINE CHART OF CASES

# In[256]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Sample DataFrame (Replace this with your actual data)
data = {
    'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
    'Confirmed': [100, 150, 200]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Animated Line Chart of Confirmed Cases Over Time'])

# Initialize an empty line chart
line_chart = go.Scatter(x=[], y=[], mode='lines')

# Add the line chart to the subplot
fig.add_trace(line_chart)

# Update layout
fig.update_layout(
    title='Animated Line Chart of Confirmed Cases Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Confirmed Cases',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Scatter(x=df['Date'][:k+1], y=df['Confirmed'][:k+1])],
                   name=f'frame{k+1}') for k in range(1, len(df))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# In[257]:


pip install matplotlib


# # RUNNING BAR CHART OF DEATHS

# In[258]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Assuming df is your DataFrame
# If not, you can read your data into a DataFrame using pd.read_csv or pd.read_excel

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Running Bar Chart of Deaths Over Time'])

# Initialize an empty bar chart
bar_chart = go.Bar(x=[], y=[])

# Add the bar chart to the subplot
fig.add_trace(bar_chart)

# Update layout
fig.update_layout(
    title='Running Bar Chart of Deaths Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Deaths',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Bar(x=df_transposed.index,
                                y=df_transposed.iloc[:, :k+1].iloc[:, -1], marker_color='red')],
                   name=f'frame{k+1}') for k in range(1, len(df_transposed.columns))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# # ANIMATED HEATMAP OF DEATHS

# In[259]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Assuming df is your DataFrame
# If not, you can read your data into a DataFrame using pd.read_csv or pd.read_excel

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Animated Heatmap of Deaths Over Time'])

# Initialize an empty heatmap
heatmap = go.Heatmap(z=[], x=[], y=[])

# Add the heatmap to the subplot
fig.add_trace(heatmap)

# Update layout
fig.update_layout(
    title='Animated Heatmap of Deaths Over Time',
    xaxis_title='Country/Region',
    yaxis_title='Date',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Heatmap(z=df_transposed.iloc[:, :k+1].values,
                                    x=df_transposed.index,
                                    y=df_transposed.columns)],
                   name=f'frame{k+1}') for k in range(1, len(df_transposed.columns))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# # ANIMATED LINE CHART OF DEATHS

# In[260]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Assuming df is your DataFrame
# If not, you can read your data into a DataFrame using pd.read_csv or pd.read_excel

# Transpose the DataFrame for easier plotting
df_transposed = df.transpose()

# Create a subplot
fig = make_subplots(rows=1, cols=1, subplot_titles=['Animated Line Chart of Deaths Over Time'])

# Initialize an empty line chart
line_chart = go.Scatter(x=[], y=[], mode='lines')

# Add the line chart to the subplot
fig.add_trace(line_chart)

# Update layout
fig.update_layout(
    title='Animated Line Chart of Deaths Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Deaths',
    template='plotly_dark'
)

# Define animation frames
frames = [go.Frame(data=[go.Scatter(x=df_transposed.columns,
                                     y=df_transposed.iloc[:, :k+1].iloc[:, -1])],
                   name=f'frame{k+1}') for k in range(1, len(df_transposed.columns))]

# Update frames and layout for animation
fig.frames = frames
fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                     buttons=[dict(label='Play',
                                                   method='animate',
                                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   fromcurrent=True,
                                                                   mode='immediate')])])])

# Show the plot
fig.show()


# # PROJECT REPORT 

# 1. ABOUT DATASET :
#    It comprises of 2 datasets : 
#    a) Confirmed : The dataset appears to contain information related to the COVID-19 pandemic, specifically the number of             confirmed cases reported on different dates across various provinces or states and countries or regions. The columns             include details such as the province/state, country/region, latitude, longitude, and daily counts of confirmed cases from       January 22, 2020, to the present date. 
#    b) Deaths : The dataset appears to contain information related to the COVID-19 pandemic, specifically the number of deaths         reported on different dates across various provinces or states and countries or regions. The columns include details such       as the province/state, country/region, latitude, longitude, and daily counts of confirmed cases from January 22, 2020, to       the present date. 
#    
#  2. OBJECTIVES : 
#     It is focused on visualizing and animating the spread of COVID-19 cases over time across different countries or regions. The     specific objectives of the project may include:
#     a) Displaying the number of COVID-19 cases, deaths, and other related metrics over time and Providing insights into the            trends and patterns of the pandemic.
#     b) Creating dynamic visualizations such as animated bar charts or heatmaps to illustrate the progression of COVID-19 cases          over time and Enhancing the user's understanding of how the situation has evolved.
#     c) Allowing users to compare the COVID-19 data between different countries or regions and Highlighting variations in the            spread and impact of the virus across locations.
#     d) Analyzing the temporal patterns of COVID-19 cases, deaths, and other factors and Identifying specific periods of                significant increase or decrease in the number of cases.
#     e) Implementing interactive features, such as play buttons, to allow users to control and explore the animation at their own        pace.
#     f) Enabling users to focus on specific time ranges or countries of interest and Insights into First Case to First Death            Duration:
#     g) Providing information on the duration between the first reported case and the first reported death in different countries        or regions.
#     
#  3. ANALYSIS AND INSIGHTS :
#    1. BAR CHART : 
#       The bar chart illustrates the frequency count of the number of days from the first reported COVID-19 case to the first           death across different countries or regions. Each bar represents a specific time interval, indicating the range of days it       took for the first death to occur after the initial case was reported. The x-axis displays the days elapsed, while the y-       axis shows the countries or regions. The chart provides a visual representation of the distribution of response times           globally, offering insights into the effectiveness of healthcare systems, government interventions, and overall                 preparedness.
#       
#       ANALYSIS : 
#       a) The bars provide a visual distribution of the number of days it took for each country or region to report its first              death after the initial case. Varied bar lengths indicate differences in the response time and effectiveness of                  healthcare systems across regions.
#       b) Some bars may have negative values on the x-axis. Negative values indicate cases where the reported death preceded the          reported case. This anomaly could be due to data recording errors or unique circumstances.
#       c) Clusters of bars or outliers may be present.Clusters suggest groups of countries with similar response times, while              outliers may signify exceptional cases requiring further investigation. 
#       d) Analyzing regional patterns can provide insights into the effectiveness of regional healthcare systems, government              responses, or other factors influencing the time from the first case to the first death.
#       
#       OBSERVATIONS : 
#       a) The chart highlights significant variability in the time it took for different countries or regions to experience the            first death after reporting the initial case.Some areas exhibit a rapid response, with a short duration between the              first case and first death, while others show a prolonged period.
#       b) Countries with a shorter duration between the first case and death may suggest a more efficient and responsive                  healthcare system.Longer durations may indicate challenges in managing and controlling the spread of the virus,                  potentially reflecting healthcare infrastructure limitations.
#       c) Regions with shorter timeframes might have implemented early and effective public health measures, such as testing,              contact tracing, and quarantine protocols.Longer durations may suggest delayed or less effective interventions,                  possibly leading to higher mortality rates.
#       d) Instances of negative values on the x-axis, indicating deaths preceding reported cases, could be attributed to data              reporting discrepancies or anomalies.Investigating such anomalies is crucial for ensuring the accuracy and reliability          of the data.
#       e) It is essential to consider data quality, completeness, and consistency when interpreting the findings, especially when          dealing with negative values or unusual patterns.
#       
#       INSIGHTS AND MANAGERIAL IMPLICATIONS :
#       a) Countries with shorter bars indicate a rapid response to the pandemic.Study strategies implemented by these countries            for potential best practices. Share insights globally for collaborative learning.
#       b) Negative values or unusually short/long bars may indicate data anomalies.Investigate anomalies to ensure data accuracy.          Rectify errors and maintain data integrity for reliable analysis.
#       c) Variations in response times may reflect the effectiveness of implemented policies.Assess the impact of policy measures          on response times. Identify successful policies and consider adjustments where needed.
#       d) Countries with longer response times may need additional resources.Consider reallocating resources, providing support,          or collaborating with international organizations to strengthen healthcare capabilities.
#       e) Clusters of similar response times may indicate regional preparedness levels.Evaluate preparedness strategies in                regions with effective responses. Share recommendations with less-prepared regions for improved readiness.
#       f) Regularly monitor and update the dataset.Implement a continuous monitoring system for real-time insights. Stay informed          about changes in response times and adapt strategies accordingly.
#       
#    2. RUNNING BAR CHART OF CASES :
#       The running bar chart over time provides a dynamic representation of the number of cases over different dates. The chart         uses an animated approach to sequentially display bars corresponding to the number of cases on each date. This                   visualization allows for a temporal understanding of how the cases evolve, offering insights into patterns and trends.
#       
#       ANALYSIS :
#       a) The animation sequentially reveals how the number of cases changes over time.Patterns such as spikes, plateaus, or              declines in case numbers become apparent.
#       b) The speed at which bars appear indicates the rate of change in case numbers.Rapid increases or decreases may suggest            critical periods or interventions.
#       c) Scrutinize any outliers or anomalies where there are unexpected spikes or drops in cases.Investigate the circumstances          surrounding these points to uncover irregularities in data reporting or identify critical events.
#       d) Overlay the chart with significant events (e.g., policy implementations, public gatherings) to assess their impact on            case numbers.Determine if specific events correlate with spikes or downturns in the running bar chart.
#       
#       OBSERVATIONS :
#       a) The running bar chart serves as a dynamic tool for continuous monitoring, providing actionable insights for decision-            makers.
#       b) Regular updates and analyses contribute to a proactive and adaptive approach to public health management.
#       c) The combination of historical trends and real-time data supports evidence-based decision-making and strengthens the              overall response to health challenges.
#       
#       INSIGHTS AND MANAGERIAL IMPLICATIONS :
#       a) Allocate resources strategically based on identified critical periods and areas with the highest case numbers.Ensure            that healthcare facilities and personnel are adequately prepared during anticipated surges.
#       b) Evaluate the impact of existing policies and interventions on the running bar chart.Adjust policies based on insights            gained from the chart to enhance their effectiveness.
#       c) Launch targeted public health campaigns during periods of rising cases to raise awareness and encourage preventive              measures.
#       d) Use historical trends from the running bar chart for forecasting future case numbers.Enhance preparedness by                    anticipating potential peaks and allocating resources accordingly.
#       e) Tailor communication strategies based on the insights gained from the running bar chart.Effectively communicate the              rationale behind interventions and the expected impact on case numbers.
#       
#   3. ANIMATED HEAT MAP FOR CASES :
#      The animated heatmap visualizes the temporal evolution of a dataset across different countries/regions over time. Each cell      in the heatmap represents a specific value, with the color intensity indicating the magnitude of that value. The animation      progresses through frames, revealing how these values change dynamically.
#      
#      ANALYSIS :
#      a) Observe how the color intensity in the heatmap changes over time, indicating temporal trends in the dataset.Identify             periods of significant fluctuation or stability.
#      b) Analyze variations in color intensity across different countries/regions.Note regions that consistently exhibit high or         low values, indicating potential areas of concern or success.
#      c) Explore patterns of correlation or inverse correlation between countries/regions.Identify clusters or groups of regions         with similar trends or divergent patterns.
#      d) Examine the heatmap for spikes or dips coinciding with specific events (e.g., policy changes, natural disasters).
#         Evaluate the impact of these events on the dataset.
#         
#      OBSERVATIONS : 
#      a) The animated heatmap highlights the dynamic nature of the dataset, showcasing how values evolve over time.Offers a real-         time view of changing patterns and trends.
#      b) Spatial disparities are evident through variations in color intensity.Regions with distinct challenges or successes             become visually apparent.
#      c) The heatmap illustrates complex interactions between temporal and geographical dimensions.Enables a comprehensive               understanding of how different regions contribute to the overall dataset.
#      d) Serves as a powerful decision support tool for managers and policymakers.Facilitates evidence-based decision-making by           providing a visual representation of data trends.
#      e) The animated heatmap can be used as a communication aid to convey complex insights to diverse stakeholders.Enhances the         ability to convey the evolving nature of the dataset.
#      
#      INSIGHTS AND MANAGERIAL IMPLICATIONS :
#      a) Allocate resources based on regions showing sustained high values, indicating potential hotspots.Enhance preparedness in         regions where values are prone to sudden changes.
#      b) Tailor policies based on insights derived from the heatmap.Implement targeted interventions in regions with specific             needs indicated by the data.
#      c) Utilize the heatmap as part of an early warning system.Identify emerging trends and take proactive measures to address           potential challenges.
#      d) Identify high-risk regions and implement risk mitigation strategies.Mitigate the impact of potential adverse events by           strategic planning informed by the heatmap.
#     
#   4. ANIMATED LINE CHART OF CASES :
#      The chart displays a temporal evolution of confirmed cases, with each point representing the cumulative number of confirmed      cases at a specific date. The x-axis denotes the date, and the y-axis represents the corresponding number of confirmed          cases. As time progresses, the line on the chart dynamically updates, providing a clear and animated representation of the      growth in confirmed cases.
#      
#      ANALYSIS :
#      a) The line chart visually represents the temporal trend of confirmed COVID-19 cases over the specified time period (from           '2022-01-01' to '2022-01-03').Each frame of the animation adds a new data point, creating a dynamic depiction of the             evolving situation.
#      b) The steepness of the line indicates the growth rate of confirmed cases. A steeper slope suggests a higher rate of               infection spread.Managers can observe periods of rapid increase or decline in cases, identifying critical points in the         timeline.
#      c) Peaks in the line represent points where the number of confirmed cases is highest. These peaks may indicate potential           outbreaks or periods of increased testing.Valleys between peaks show periods of relatively lower case counts, providing         insights into potential control measures or interventions.
#      d) Changes in the direction or slope of the line may highlight significant events or interventions that impact the spread           of the virus.
#      
#      OBSERVATIONS :
#      a) Managers can identify overall trends in the spread of the virus, helping to understand whether the situation is                 improving, stabilizing, or worsening.
#      b) The animated line chart serves as a valuable decision support tool for public health officials, aiding in strategic             decision-making.
#      c) By closely monitoring the line chart, managers can assess the outcomes of interventions and adjust strategies                   accordingly.
#      d) Use the chart to enhance public awareness and understanding of the ongoing situation, promoting adherence to recommended         guidelines.
#      
#      INSIGHTS AND MANAGERIAL IMPLICATIONS :
#      a) Managers can use the chart to anticipate peaks and allocate healthcare resources more effectively during periods of high         case counts.Resource planning may involve adjusting the number of hospital beds, ventilators, and medical staff based on         the observed trends.
#      b) Evaluate the effectiveness of implemented policies or interventions by correlating changes in the trend with specific           policy dates.Adjust public health measures based on the observed impact of interventions on the growth rate.
#      c) Use the animated chart as a visual communication tool for stakeholders, policymakers, and the public.Highlight key               points in the timeline to communicate the severity of the situation and the effectiveness of response efforts.
#      d) The chart aids in preparedness planning for potential future waves or spikes in confirmed cases.Emergency response plans         can be refined based on insights gained from the visualization.
#      
#   5. RUNNING BAR CHART FOR DEATHS : 
#      The running bar chart visually depicts the progression of deaths over time in response to a dynamic dataset. Each bar            represents the cumulative number of deaths at a specific point in time. The animation iteratively adds bars to showcase the      evolving trend of mortality.
# 
#      ANALYSIS : 
#      a) Observe the temporal patterns in the running bar chart to identify periods of rising and falling mortality rates. Peaks         may indicate critical phases of the health crisis.
#      b) Analyze the spacing between bars to understand whether the mortality rate is accelerating, decelerating, or remaining           relatively stable over time.
#      c) Identify sudden spikes in the chart, as they may indicate localized outbreaks or periods of increased vulnerability.
#      d) Peaks in the running bar chart signify critical time points with elevated mortality rates. Understanding the events             around these peaks provides insights into the severity of the crisis at specific moments.
#      e) Look for cyclical patterns or seasonal variations in mortality. Certain health issues might be more prevalent during             specific times of the year.
#      f) Compare the running bar chart across different periods to assess the impact of interventions. Evaluate whether                   implemented measures effectively mitigated mortality.
#      
#      OBSERVATIONS :
#      a) The running bar chart provides a visually impactful representation of mortality trends over time. The animation captures         attention and facilitates a nuanced understanding of the evolving situation.
#      b) Managers and policymakers can make informed decisions by closely monitoring the chart. The visual nature aids in                 comprehending the pace and severity of the crisis.
#      c) The dynamic nature of the chart emphasizes the need for real-time adaptation in response strategies. Flexibility and             adaptability are crucial components of effective crisis management.
#      d) Use the insights gained from the running bar chart to continuously improve response mechanisms. Learn from patterns and         apply lessons to enhance future crisis management.
#      
#      INSIGHTS AND MANAGERIAL IMPLICATIONS :
#      a) Allocate resources strategically by anticipating peak periods. Ensure that healthcare facilities are adequately                 equipped during times of increased mortality.
#      b) Tailor response strategies based on the dynamic nature of the running bar chart. Adjust interventions in real-time to           address emerging trends.
#      c) Use the running bar chart as an early warning system. Sudden spikes can trigger immediate responses and targeted                 interventions to contain outbreaks.
#      d) Utilize the chart in public communication to transparently convey the progression of mortality. This builds trust and           helps the public understand the evolving nature of the health crisis.
#     
#   6. ANIMATED HEATMAP FOR DEATHS : 
#      The Animated Heatmap visually represents the progression of deaths over time across different countries or regions. The          heatmap utilizes color intensity to depict the varying degrees of mortality, with each cell representing the intersection        of a specific country/region and date. As the animation unfolds, viewers can observe dynamic changes in the intensity of        color, offering a comprehensive view of the evolving mortality patterns.
#      
#      ANALYSIS :
#      a) Identification of Peaks and Valleys: The heatmap enables the identification of peaks, valleys, and trends in mortality           over different time periods. Peaks may indicate significant events such as outbreaks or specific time frames of                 increased mortality.
#      b) Variations in color intensity highlight regions experiencing higher or lower mortality rates. Managers can pinpoint             specific countries or regions with consistent issues or improvements in mortality.
#      c) Abrupt changes in color intensity may signify potential outbreaks or critical periods. These anomalies can be                   investigated further to understand the causes and implement timely interventions.
#      d) Analyzing the animation over an extended period allows for a comprehensive understanding of mortality trends. Managers           can identify recurring patterns, contributing factors, and potential seasonality.
#      e) Differences in color intensity across regions offer insights into regional health disparities. Managers can explore the         factors influencing these variations and tailor interventions accordingly.
#      
#      OBSERVATIONS :
#      a) The animation's dynamic nature offers a real-time understanding of mortality dynamics.Stakeholders can observe changes           and react promptly to evolving situations.
#      b) The animated heatmap can be a powerful communication tool. Its visual nature makes complex data accessible to a broader         audience, facilitating transparent communication with the public, healthcare professionals, and policymakers.
#      c) Continuous observation of the animated heatmap allows for adaptive strategies. Managers can continuously learn from             emerging patterns and adjust their approaches, ensuring a dynamic and responsive healthcare strategy.
#      
#      INSIGHTS AND MANAGERIAL IMPLICATIONS :
#      a) Targeted Interventions: Identify regions or countries with sustained high mortality for targeted resource allocation.           This proactive approach ensures that resources are efficiently distributed to areas in most need.
#      b) Recognizing patterns associated with increased mortality allows managers to implement preventive measures in advance.           This could include vaccination campaigns, public health awareness, or enhanced medical infrastructure.
#      c) The heatmap serves as a decision support tool for policymakers. Informed decisions can be made based on real-time               insights into mortality dynamics, guiding the formulation of effective healthcare policies.
#      
#   7. ANIMATED LINE CHART FOR DEATHS :
#      This animated line chart visually represents the progression of deaths over time, offering dynamic insights into mortality      trends. The chart showcases a line that evolves with each frame, where each frame corresponds to a specific time interval.      The animation allows viewers to observe the changing patterns of deaths, identifying peaks, troughs, and overall trends.
#      
#      ANALYSIS : 
#      a) The animated line chart allows for the identification of trends in mortality over different time periods. Peaks and             valleys indicate periods of increased or decreased mortality.
#      b) The dynamic nature of the line chart provides real-time insights into mortality dynamics, enabling stakeholders to               observe changes as they occur.
#      c) Differences in the line chart's trajectories across regions highlight regional disparities in mortality rates.
#      d) Abrupt changes or spikes in the line chart may signify critical periods requiring closer examination.
#      
#      OBSERVATIONS : 
#      a) The animated line chart offers a real-time understanding of mortality dynamics, allowing stakeholders to observe changes         and respond promptly.
#      b) The animated line chart serves as a powerful communication tool, translating complex data into accessible visual                 insights.
#      c) Continuous observation of the animated line chart supports adaptive strategies.
#      
#      INSIGHTS AND MANAGERIAL IMPLICATIONS :
#      a) Regions or countries with sustained high mortality rates can be identified for targeted resource allocation.Proactive           resource allocation ensures that critical areas receive adequate resources, optimizing the impact of interventions.
#      b) Recognizing patterns associated with increased mortality enables the implementation of preventive measures in                   advance.This proactive approach includes vaccination campaigns, public health awareness, and strengthening medical               infrastructure in anticipation of potential challenges.
#      c) The animated line chart serves as a decision support tool for policymakers.Informed decisions based on real-time                 insights into mortality dynamics guide the formulation of effective healthcare policies, enhancing overall public               health.
#      
#  4. MANAGERIAL IMPLICATIONS :
#     The managerial implications derived from the visualizations emphasize the need for dynamic, data-driven decision-making,         proactive interventions, and collaborative efforts are given below :
#  
#     a) The visualizations enable the identification of regions or periods with higher mortality, guiding resource                      allocation.Managers can dynamically allocate resources based on real-time observations, ensuring efficient use of                resources where they are most needed.
#     b) Identification of trends and critical periods allows for proactive interventions.Managers can implement preventive              measures, public health campaigns, and infrastructure enhancements in advance, reducing the impact of potential health          crises.
#     c) Real-time understanding of mortality dynamics supports adaptive policy formulation.Policymakers can adjust healthcare            policies dynamically, responding to emerging trends and ensuring that policies remain effective and relevant over time.
#     d) Recognition of regional disparities informs tailored interventions.Managers can design equitable healthcare strategies,          addressing specific challenges in different regions and promoting inclusive access to quality healthcare.
#     e) Visualizations serve as powerful communication tools.Transparent communication with the public, healthcare professionals,        and policymakers becomes more effective, fostering trust, understanding, and collaborative efforts.
#     f) The visualizations act as decision support tools for policymakers.Informed decisions based on real-time insights into            mortality dynamics guide policymakers in formulating effective healthcare policies, ensuring a responsive and evidence-          driven approach.
#     g) The dynamic nature of the visualizations supports continuous learning. Managers and policymakers can continuously adapt          strategies based on emerging patterns, promoting a culture of learning and responsiveness in the healthcare system.
#     h) Early identification of critical periods enhances preparedness.Public health agencies can enhance preparedness for              potential outbreaks, emergencies, or increased mortality, minimizing the impact on public health.
#     
#  5. RECOMMENDATIONS :
#     By implementing these recommendations, we can strengthen healthcare systems, enhance responsiveness, and contribute to           improved health outcomes on a local and global scale are given below :
#     
#     a) Integrate data sources for a comprehensive understanding of health dynamics.Comprehensive data integration enables a more        accurate representation of health trends and facilitates informed decision-making.
#     b) Develop real-time data analytics capabilities.Real-time analytics support timely decision-making, allowing for quick            responses to emerging health challenges and trends.
#     c) Invest in predictive modeling tools.Predictive modeling helps forecast potential health issues, allowing for proactive          planning and resource allocation.
#     d) Develop early warning systems for disease outbreaks.Early warning systems enable rapid response to potential outbreaks,          minimizing their impact on public health.
#     e) Facilitate collaboration between health agencies.Interagency collaboration ensures a unified and coordinated response to        health challenges, leveraging diverse expertise and resources.
#     f) Strengthen health information systems.Robust information systems facilitate data collection, sharing, and analysis,              supporting evidence-based decision-making.
#     g) Tailor healthcare strategies to regional needs.Regionalized strategies acknowledge diverse health challenges, allowing          for targeted interventions and resource allocation.
#     h) Invest in effective public health communication.Transparent communication builds trust, encourages public adherence to          health guidelines, and facilitates collaboration between healthcare professionals and the public.
#     i) Provide continuous training for healthcare professionals.Continuous training ensures that healthcare professionals are          equipped to handle evolving health challenges and utilize advanced analytical tools.
#     
#     REFERENCE : FEW CODES FROM CHATGPT

# In[ ]:


)

