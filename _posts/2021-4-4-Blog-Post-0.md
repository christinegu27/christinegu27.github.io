---
layout: post
title: Blog Post 0
---

## The Palmer Penguins Dataset (again)

### Getting the Data

The imported packages used to create a visualization of the Palmer Penguins dataset will be `matplotlib` and `pandas`. `pandas` is needed to read in the data and will be used to manipulate the dataset and prepare it for graphing. The tools from `matplotlib` will be used to actually create the graph. First, let's read in the data from the internet and take a look at a couple rows.

```python
import pandas as pd
from matplotlib import pyplot as plt

url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
penguins.head()
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
      <th>studyName</th>
      <th>Sample Number</th>
      <th>Species</th>
      <th>Region</th>
      <th>Island</th>
      <th>Stage</th>
      <th>Individual ID</th>
      <th>Clutch Completion</th>
      <th>Date Egg</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
      <th>Sex</th>
      <th>Delta 15 N (o/oo)</th>
      <th>Delta 13 C (o/oo)</th>
      <th>Comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAL0708</td>
      <td>1</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N1A1</td>
      <td>Yes</td>
      <td>11/11/07</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>MALE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Not enough blood for isotopes.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAL0708</td>
      <td>2</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N1A2</td>
      <td>Yes</td>
      <td>11/11/07</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>FEMALE</td>
      <td>8.94956</td>
      <td>-24.69454</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAL0708</td>
      <td>3</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N2A1</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>FEMALE</td>
      <td>8.36821</td>
      <td>-25.33302</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAL0708</td>
      <td>4</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N2A2</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Adult not sampled.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAL0708</td>
      <td>5</td>
      <td>Adelie Penguin (Pygoscelis adeliae)</td>
      <td>Anvers</td>
      <td>Torgersen</td>
      <td>Adult, 1 Egg Stage</td>
      <td>N3A1</td>
      <td>Yes</td>
      <td>11/16/07</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>FEMALE</td>
      <td>8.76651</td>
      <td>-25.32426</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

The groupby function from `pandas` will automatically split up the dataset by the chosen parameter, which is penguin species in this case. The goal of this blog post will be to create a color-coded histogram showing the body mass of the penguins. Each penguin species will get its own color, so penguin size can be compared between species. 

```python
penguins.groupby('Species').mean().drop(['Sample Number'],axis=1)
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
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
      <th>Delta 15 N (o/oo)</th>
      <th>Delta 13 C (o/oo)</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adelie Penguin (Pygoscelis adeliae)</th>
      <td>38.791391</td>
      <td>18.346358</td>
      <td>189.953642</td>
      <td>3700.662252</td>
      <td>8.859733</td>
      <td>-25.804194</td>
    </tr>
    <tr>
      <th>Chinstrap penguin (Pygoscelis antarctica)</th>
      <td>48.833824</td>
      <td>18.420588</td>
      <td>195.823529</td>
      <td>3733.088235</td>
      <td>9.356155</td>
      <td>-24.546542</td>
    </tr>
    <tr>
      <th>Gentoo penguin (Pygoscelis papua)</th>
      <td>47.504878</td>
      <td>14.982114</td>
      <td>217.186992</td>
      <td>5076.016260</td>
      <td>8.245338</td>
      <td>-26.185298</td>
    </tr>
  </tbody>
</table>
</div>

### Making the Visualization

To start, create the figure and axes objects that will be used to graph the data. This figure will also need a title and an x-axis label (since this is a histogram, the y-axis label isn't necessary). In order to add these labels, the set function will be used with the `ax` object.

```python
fig, ax = plt.subplots()

ax.set(xlabel= "Penguin Body Mass (g)", title = "Penguin Body Mass by Species")
```
![]({{christinegu27.github.io}}/images/empty penguin graph.png)

Before any data can be plotted on the graph, a plotting function needs to be defined. This function will be applied to each group, or penguin species, to allow the histogram to color code by species.  This will allow the histogram  to color code by  species. The plotting function will be defined to also give labels for each group it is plotting and provide a legend.

```python
def plot_hist(df, colname, alpha):
    """
    Plots a histogram of the given data
    parameter df: the dataset containing the data to be plotted
    parameter colname: the column of data to be plotted
    parameter alpha: transparency for aesthetics
    """
    #gets the current species being plotted
    species = df['Species'].iloc[0]
    #uses the histogram plotting function to create a histogram of the penguin mass
    #also assigned a label to the plotted data
    ax.hist(df[colname], alpha=alpha, label = species.split()[0])
    #displays the legend
    ax.legend()
```
Now that the function needed to create the histogram is defined, it's finally time to create the visualization. The groupby function will be used on the dataset to split it up by penguin species. Then for each species, the apply function will literally apply the plotting function to the data. `plot_hist` will be called 3 times in total, one for each penguin species.

```python
penguins.groupby(['Species']).apply(plot_hist, 'Body Mass (g)', .5)
fig
```
![]({{christinegu27.github.io}}/images/blog post 0 penguin.png)

Our final visualization shows the histogram of penguin body mass by species. The legend shows the color corresponding with each penguin species, where green represents Gentoo penguins, blue (Adelie) and orange (Chinstrap) penguins. On average, Adelie and Chinstrap penguins are about the same size, but Gentoo penguins are generally larger.

{::options parse_block_html="true" /}
<div class="got-help">
What I learned from peer feedback: 
</div>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<div class="gave-help">
Suggestions given: 
</div>
{::options parse_block_html="false" /}