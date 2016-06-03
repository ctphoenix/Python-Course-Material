---
title       : Module 4 (Whisky) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
-  Here we provide a basic demonstration of an interactive grid plot using Bokeh.  Execute the following code, and follow along the comments. We will later adapt this code to plot the correlations among distillery flavor profiles, as well as a geographical map of distilleries colored by region and flavor profile.

*** =hint
- 

*** =pre_exercise_code
```{python}
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
model = SpectralCoclustering(n_clusters=6, random_state=0)
flavors = whisky.iloc[:,2:14] # extract flavor attributes
corr_whisky = pd.DataFrame.corr(flavors.transpose())
model.fit(corr_whisky)
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
```

*** =sample_code
```{python}
# First, we import a tool to allow text to pop up on the plot when the cursor
# hovers over it.  Also, we import a data structure used to store definitions
# of what to plot in Bokeh.  Finally, we will use numpy for this section as well!

from bokeh.models import HoverTool, ColumnDataSource
import numpy as np

# Let's plot a simple 5x5 grid of squares, alternating in color as red and blue.

plot_values = [1,2,3,4,5]
plot_colors = ["red", "blue"]

# How do we tell Bokeh to plot each point in a grid?  Let's use a function that
# finds each combination of values from 1-5.
from itertools import product

grid = list(product(plot_values, plot_values))
print(grid)

# The first value is the x coordinate, and the second value is the y coordinate.
# Let's store these in separate lists.

xs, ys = zip(*grid)
print(xs)
print(ys)

# Now we will make a list of colors, alternating between red and blue.

colors = [plot_colors[i%2] for i in range(len(grid))]
print(colors)

# Finally, let's determine the strength of transparency (alpha) for each point,
# where 0 is completely see-through.

alphas = np.linspace(0, 1, len(grid))

# Bokeh likes each of these to be stored in a special dataframe, called
# ColumnDataSource.  Let's store our coordinates, colors, and alpha values.

source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
        "alphas": alphas,
    }
)
# We are ready to make our interactive Bokeh plot!

fig = figure(tools="resize, hover, save")
fig.rect("x", "y", 0.9, 0.9, source=source, color="colors",alpha="alphas")
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Value": "@x, @y",
    }
show(fig)
```

*** =solution
```{python}
# First, we import a tool to allow text to pop up on the plot when the cursor
# hovers over it.  Also, we import a data structure used to store definitions
# of what to plot in Bokeh.  Finally, we will use numpy for this section as well!

from bokeh.models import HoverTool, ColumnDataSource
import numpy as np

# Let's plot a simple 5x5 grid of squares, alternating in color as red and blue.

plot_values = [1,2,3,4,5]
plot_colors = ["red", "blue"]

# How do we tell Bokeh to plot each point in a grid?  Let's use a function that
# finds each combination of values from 1-5.
from itertools import product

grid = list(product(plot_values, plot_values))
print(grid)

# The first value is the x coordinate, and the second value is the y coordinate.
# Let's store these in separate lists.

xs, ys = zip(*grid)
print(xs)
print(ys)

# Now we will make a list of colors, alternating between red and blue.

colors = [plot_colors[i%2] for i in range(len(grid))]
print(colors)

# Finally, let's determine the strength of transparency (alpha) for each point,
# where 0 is completely see-through.

alphas = np.linspace(0, 1, len(grid))

# Bokeh likes each of these to be stored in a special dataframe, called
# ColumnDataSource.  Let's store our coordinates, colors, and alpha values.

source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
        "alphas": alphas,
    }
)
# We are ready to make our interactive Bokeh plot!

fig = figure(tools="resize, hover, save")
fig.rect("x", "y", 0.9, 0.9, source=source, color="colors",alpha="alphas")
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Value": "@x, @y",
    }
show(fig)
```

*** =sct
```{python}
test_function("",
              not_called_msg = "Make sure to call ``!",
              incorrect_msg = "Check your definition of `` again.")
test_object("",
            undefined_msg = "Did you define ``?",
            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!")
```
