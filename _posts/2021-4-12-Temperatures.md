---
layout: post
title: Analyzing a Temperature Dataset
---

This blog post will focus on visualizing temperature data measured over the past few decades.
First, the required packages will be imported. As usual, `pandas` will be used to process the data, and `sqlite3` is needed to navigate the database that will be created. Linear regression will be used to estimate the average temperature increase so `sklearn.linear_model` is imported, and finally `plotly` allows for the creation of interactive scatterplots.


```python
import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from plotly import express as px
```

The temps.csv file is so large that it must be downloaded on the computer, instead of loading it in from the internet like the other two files.


```python
temperatures = pd.read_csv("temps.csv")
stations = pd.read_csv("https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv")
country = pd.read_csv("https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv")
temperatures.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>VALUE1</th>
      <th>VALUE2</th>
      <th>VALUE3</th>
      <th>VALUE4</th>
      <th>VALUE5</th>
      <th>VALUE6</th>
      <th>VALUE7</th>
      <th>VALUE8</th>
      <th>VALUE9</th>
      <th>VALUE10</th>
      <th>VALUE11</th>
      <th>VALUE12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>-89.0</td>
      <td>236.0</td>
      <td>472.0</td>
      <td>773.0</td>
      <td>1128.0</td>
      <td>1599.0</td>
      <td>1570.0</td>
      <td>1481.0</td>
      <td>1413.0</td>
      <td>1174.0</td>
      <td>510.0</td>
      <td>-39.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1962</td>
      <td>113.0</td>
      <td>85.0</td>
      <td>-154.0</td>
      <td>635.0</td>
      <td>908.0</td>
      <td>1381.0</td>
      <td>1510.0</td>
      <td>1393.0</td>
      <td>1163.0</td>
      <td>994.0</td>
      <td>323.0</td>
      <td>-126.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1963</td>
      <td>-713.0</td>
      <td>-553.0</td>
      <td>-99.0</td>
      <td>541.0</td>
      <td>1224.0</td>
      <td>1627.0</td>
      <td>1620.0</td>
      <td>1596.0</td>
      <td>1332.0</td>
      <td>940.0</td>
      <td>566.0</td>
      <td>-108.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1964</td>
      <td>62.0</td>
      <td>-85.0</td>
      <td>55.0</td>
      <td>738.0</td>
      <td>1219.0</td>
      <td>1442.0</td>
      <td>1506.0</td>
      <td>1557.0</td>
      <td>1221.0</td>
      <td>788.0</td>
      <td>546.0</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1965</td>
      <td>44.0</td>
      <td>-105.0</td>
      <td>38.0</td>
      <td>590.0</td>
      <td>987.0</td>
      <td>1500.0</td>
      <td>1487.0</td>
      <td>1477.0</td>
      <td>1377.0</td>
      <td>974.0</td>
      <td>31.0</td>
      <td>-178.0</td>
    </tr>
  </tbody>
</table>
</div>

We can see that the temperatures dataframe has columns with the temperature for each month of the year. In order to prepare the data for later functions, all of these columns will be stacked into one "Month" column using the `stack` function from `pandas`  before being added to the database. Other edits include giving the new columns to more informative names and data types. A final column called "FIPS_10-4" is created to identify which country each station is located in.


```python
#taken from Week 2 Monday lecture
def prepare_df(df):
    """
    Prepares the dataframe by stacking columns into one "Month" column, 
    editing the temperature values, and 
    adding a column for identifying station country code.
    parameter df: the dataframe to be modified
    returns the processed dataframe 
    """
    df = df.set_index(keys=["ID", "Year"])
    df = df.stack()
    df = df.reset_index()
    #gives more informative column names
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    #changes the Month column from integers to string
    df["Month"] = df["Month"].str[5:].astype(int)
    #puts temperature values in more understandab;e values
    df["Temp"]  = df["Temp"] / 100
    #creates a country code from the ID column for merging later on
    df["FIPS_10-4"] = df["ID"].str[0:2]
    return(df)
```
SQL is unable to process variables with spaces in the name without special workarounds, so the columns in the country dataframe will be renamed to avoid the hassle.  

```python
country = country.rename(columns={'FIPS 10-4': 'FIPS_10-4', 'ISO 3166': 'ISO_3166'})
```

Now that the preparation has finished, it's time to create the database. The database will have three tables: temperatures, countries, and stations to match with each data set. First a connection to a database will be created, and then the data will be added using the `to_sql` function conveniently included with `pandas`. 

Since the temps.csv file is so large, it has to be divided into sections, where each section will be processed by the `prepare_df` function, then loaded in. However stations and countries are small enough and can be loaded directly from the dataframes previously created. After running this code, a new Data Base file called "temps" will appear in the computer. 


```python
conn = sqlite3.connect("temps.db")

df_iter = pd.read_csv("temps.csv", chunksize = 100000)
for df in df_iter:
     df = prepare_df(df)
     df.to_sql("temperatures", conn, if_exists = "append", index = False)
    
stations.to_sql("stations", conn, if_exists = "replace", index = False)
country.to_sql("countries", conn, if_exists = "replace", index = False)
```

A cursor will be created to navigate the temps database that was just created. The first task will be to check if the data has been added as intended by looking at the column names of each table.


```python
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

#will print out the variable names and types for each of the 3 tables in the database
for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "temperatures" (
    "ID" TEXT,
      "Year" INTEGER,
      "Month" INTEGER,
      "Temp" REAL,
      "FIPS_10-4" TEXT
    )
    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT
    )
    CREATE TABLE "countries" (
    "FIPS_10-4" TEXT,
      "ISO_3166" TEXT,
      "Name" TEXT
    )
    

These names match the columns in the dataframes from before, so it looks like there weren't any issues. Now that we've finished creating the database, it's importance to close the connection.

```python
conn.close()
```

### Using the Database

We can do some cool things with this database. For example, let's write a function that will give the temperature readings for a specific country over a certain amount of years. A SQL command will allow us to search through temps and pick out the data that matches our conditions, instead of having to filter one giant dataframe. The SQL commands are translated as follows:
* `SELECT`: Chooses the relevant variables. Each variable has a capital letter at the beginning denoting which of the 3 tables it is located in.
* `FROM`: Designates which table to take the data from. Temperatures, stations, and countries are abbreviated for readability
* `LEFT JOIN`: Combines two tables on a specified column that appears in both
* `WHERE`: Filters out the data through a set of conditions. 


```python
def query_climate_database(country, year_begin, year_end, month):
    '''
    A function that returns the temperature reading in a country during a specific month
    over a specified time frame
    parameter country: the name of the country where the temperature data was taken
    parameter year_begin: an integer giving the earliest year of data to be returned
    parameter year_end: an integer giving the last year of data to be returned
    month: an integer specifiying which month should the data be taken from
    returns a dataframe with the data of stations in the country and month for 
    each of the years within the range of year_begin to year_end
    '''
    
    #reopening the connection
    conn = sqlite3.connect("temps.db")
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.name AS Country, T.Year, T.month, T.temp
    FROM temperatures T
    LEFT JOIN countries C ON T.`FIPS_10-4` = C.`FIPS_10-4`
    LEFT JOIN stations S ON T.id = S.id
    WHERE C.name = '{country}' 
        AND T.year >= {year_begin} AND T.year <= {year_end}
        AND T.month = {month}
    """.format(country=country, year_begin=year_begin, year_end=year_end, month=month)
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```

```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>

By using the `query_climate_database` function, we were able to get a dataframe showing stations in India, the latitude and longitude of each station, and the temperatures measured at these stations in January from 1980 to 2020 (whenever data was present).

### Visualizing Changes in Temperature within a Country

This section will be focused on answering the following question.

> How does the average yearly change in temperature vary in a given country?

We will use linear regression to get an estimate of the annual change in temperature for a given station. More specifically, the resulting coefficient tied to the `year` variable will show the change in temperature over time. After the coefficients for each station are collected, they will be put on an interactive scatterplot. This scatterplot will be created with Plotly Express, where moving the cursor over a point will give the station name, location, and the annual temperature change measured. By looking at this visualization, we will be able to see which stations have a larger temperature change, as well as where temperatures are increasing or actually decreasing in a country.

```python
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    """
    Creates an interactive scatterplot showing the average annual temperature change at
    weather stations
    parameter country: the name of the country where the temperature data was taken
    parameter year_begin: an integer giving the earliest year of data to be returned
    parameter year_end: an integer giving the last year of data to be returned
    month: an integer specifiying which month should the data be taken from
    parameter min_obs: mininum number of years of data needed for a given station
    returns a figure containing the map
    """
    #grabs the data from the database for the specified country, time range and month
    df = query_climate_database(country, year_begin, year_end, month)
    #counts the amount of years measured for a station
    df["Years Measured"] = df.groupby(["NAME"])["Temp"].transform(len)
    #filters out stations that don't have enough years measured
    df = df[df["Years Measured"] >= min_obs]
    #gets the annual temperature change at each station
    annual_change = df.groupby(['NAME']).apply(coef).reset_index().rename(
        columns={0:"Estimated Annual \u0394\u2103"})
    #merges the annual temperature change onto the dataset
    df = pd.merge(df, annual_change, on = ["NAME"])
    
    #create the scatterplot, any kwargs for the plot go here
    fig = px.scatter_mapbox(df, lat = "LATITUDE", lon = "LONGITUDE", 
                            hover_name = "NAME", 
                            color = "Estimated Annual \u0394\u2103",
                            color_continuous_midpoint= 0, **kwargs)
    return fig

def coef(data_group):
    """
    Helper function that will give the estimated change in yearly temperature
    using linear regression
    paramter data_group: the data to fit the model on
    returns the coefficient attached to year rounded to 4 decimal points
    """
    x = data_group[["Year"]] #predictor variable
    y = data_group["Temp"] #target variable
    LR = LinearRegression()
    LR.fit(x, y)
    return round(LR.coef_[0], 4)
```

To match with the dataset produced earlier, let's see the temperature change among stations in India from 1980 to 2020. Each station will need at least 10 years of measured temperatures to be included in the map to reduce any abnormal data points.


```python
color_map = px.colors.diverging.balance #choosing a colormap

fig = temperature_coefficient_plot("Brazil", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map,
                                   title = "Estimates of annual temperature increase in Brazil from 1980 to 2020.")

#write to html to save figure for inclusion in blog
from plotly.io import write_html
write_html(fig, "temp_country.html")

fig.show()
```
{% include temp_country.html %}

From this map, we can see that the temperature is increasing at the vast majority of stations in Brazil over the past few decades. As a sidenote, we also learned that most of the stations in the country are located closer to the coast, and not as many are in the Amazon rainforest. If somebody was curious, they could use the functions defined earlier to similarly visualize temperature changes in other countries, such as China or Australia, or over a different time frame. 

### Annual Temperature Pattern in a Country

In the stations table, the database keeps track of the longitude and latitude of each station, as well as the elevation. Let's now take a look at how elevation factors into the temperature in a given location.

> Does elevation affect the monthly temperature in a country? 

To answer this question, we will create a graph that compares the temperature across a given year between stations at high altitude and stations at lower altitude. For simplicity, "high altitude" will be any station that has an elevation above the median elevation of all stations in the country. Median to prevent any extreme outliers from skewing the data.


```python
def query_climate_database_elevation(country, year):
    '''
    Searches the database for the temperature reading from all stations in a country during 
    a specific year
    parameter country: the name of the country where the temperature data was taken
    parameter year: the year of data to be analyzed
    returns a dataframe with the temperature readings 
    '''
    
    #grabs desired data from database
    conn = sqlite3.connect("temps.db")
    cmd = \
    """
    SELECT S.name, S.stnelev AS Elevation, C.name AS Country, T.year, T.month, T.temp
    FROM temperatures T
    LEFT JOIN countries C ON T.`FIPS_10-4` = C.`FIPS_10-4`
    LEFT JOIN stations S ON T.id = S.id
    WHERE C.name = '{country}' 
        AND T.year == {year}
    """.format(country=country, year=year)
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```


```python
def temperature_elevation(country, year, **kwargs):
    """
    Creates a box plot showing temperature readings across a year for a
    country, splitting stations by their elevation
    parameter country: the name of the country where the temperature data was taken
    parameter year: the year of data to be analyzed
    returns a figure containing the box plot
    """
    df = query_climate_database_elevation ("Brazil", 2010)
    
    #finds the median of station elevation
    med_elev = df["Elevation"].median()
    
    #creates a qualitative column indicating high or low altidude station
    df['Altitude'] = "Low"
    #any stations above median elevation are "high" altitude
    df.loc[df["Elevation"] > med_elev, "Altitude"] = "High"
    
    #plots the temperature readings into a box plot faceted by altitude
    fig = px.box(df,
                 x = "Month", y = "Temp",
                 labels = {"Temp": "Temperature \u2103"},
                 hover_name = "NAME",
                 facet_col = "Altitude", **kwargs)
    #manually edits the number of ticks on x-axis
    fig.update_xaxes(nticks=12)
    return fig
```


```python
import plotly.io as pio

#sets a theme for plotly graphs
pio.templates.default = "seaborn"

fig = temperature_elevation("Brazil", 2010, 
                            title = "Temperature across Brazil in 2010 at stations with low and high elevation.")

fig.show()
```

{% include temp_elevation.html %}

We can see the relationship between temperature and elevation in this faceted box graph. In general, stations at low elevation show warmer temperatures than those at high elevation. The box plot also tells us that temperatures are coolest during the months of June, July, and August, which is the opposite of what's expected in the Northen Hemisphere.

### Annual Temperature Change per Country

The map above shows how the temperature is chaning within a country, but how does this look on a global scale? There are too many stations across the planet to follow the same steps exactly, but a similar method will be used to answer:
> How does the average annual temperature change vary globally?

The function defined below is similar to `query_climate_database`, except it is no longer limited to stations in just one country. It uses the `coef` helper function to get the estimated annual change in temperature per station, and directly returns a dataframe containing the coefficients for each station, instead of temperature data per year. Since we will eventually be creating a choropleth of countries, the latitude and longitude of each station is also no longer needed.


```python
def query_climate_database_global(year_begin, year_end, month):
    '''
    Returns a database with the estimated change in annual temperature using
    linear regression for each station.
    parameter year_begin: an integer giving the earliest year of temperature data for LR
    parameter year_end: an integer giving the last year of temperature data for LR
    month: an integer specifiying which month should the data be taken from
    returns a dataframe with the country and estimated annual change in temperature 
    for each station
    '''
    
    #gets the desired data from the database 
    conn = sqlite3.connect("temps.db")
    cmd = \
    """
    SELECT S.name, C.name as Country, T.year, T.temp
    FROM temperatures T
    LEFT JOIN countries C ON T.`FIPS_10-4` = C.`FIPS_10-4`
    LEFT JOIN stations S ON T.id = S.id
    WHERE T.year >= {year_begin} AND T.year <= {year_end}
        AND T.month = {month}
    """.format(year_begin=year_begin, year_end=year_end, month=month)
    df = pd.read_sql_query(cmd, conn)
    
    #uses linear regression to estimate annual temperature change per station
    df = df.groupby(['NAME', 'Country']).apply(coef).reset_index().rename(
       columns={0:"Estimated Annual \u0394\u2103"})
    conn.close()
    return df
```

The following lines of code import the geographic data to create the choropleth. The `countries_gj` file contains information about each country, including its borders in coordinates needed to plot country lines.


```python
from urllib.request import urlopen
import json

countries_gj_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/countries.geojson"

with urlopen(countries_gj_url) as response:
    countries_gj = json.load(response)
```


```python
def coefficient_by_country(year_begin, year_end, month,**kwargs):
    """
    Creates a choropleth of the estimated annual change in temperature by country
    parameter year_begin: an integer giving the earliest year of data to use in LR
    parameter year_end: an integer giving the last year of data to use in LR
    month: an integer specifiying which month should the data be taken from
    returns a figure with the choropleth
    """
    
    df=query_climate_database_global(year_begin = 1960, 
                       year_end = 2020,
                       month = 2)
    
    #calculates the average change in temperature across stations per country
    df=df.groupby("Country")[["Estimated Annual \u0394\u2103"]].mean().reset_index()
    #rounds average to 4 decimal places
    df["Estimated Annual \u0394\u2103"]=df["Estimated Annual \u0394\u2103"].round(4)
    #create the choropleth with the geoJSON file
    fig = px.choropleth(df, 
                    geojson=countries_gj,
                    locations = "Country",
                    locationmode = "country names",
                    color = "Estimated Annual \u0394\u2103",
                    color_continuous_midpoint = 0, range_color = (-0.25, 0.25),
                    **kwargs)
    return fig
```
```python
fig=coefficient_by_country(1960, 2020, 4, 
                           color_continuous_scale=color_map,
                           title = "Estimates of annual temperature increase from 1980 to 2020 per country.")
fig.show()
```
{% include temp_choropleth.html %}

While there are some minor differences, we see that on average, countries are seeing increasing temperatures of about 0 to 0.1 degrees Celsius each year. This seems to be independent of location (climate), since both countries close to and far from the equator are experiencing the same increase. However according to the map, there are a few countries that are actually cooling. This may be true in real life, or it could be a result of insufficient data (not enough stations or years measured).
