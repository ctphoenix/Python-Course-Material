---
title       : Homework - Visualizing Whisky Classification
description : In this homework, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.
attachments :
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In this homework, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
-  Here we provide a basic demonstration of an interactive grid plot using Bokeh.  Execute the following code, and follow along the comments. We will later adapt this code to plot the correlations among distillery flavor profiles, as well as a geographical map of distilleries colored by region and flavor profile.

*** =hint
- Just execute and read along with the code given!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv(data_filepath + "whiskies.txt")
whisky["Region"] = pd.read_csv(data_filepath + "regions.txt")
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

output_file("Basic_Example.html", title="Basic Example")
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

output_file("Basic_Example.html", title="Basic Example")
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
	test_object("fig",
            undefined_msg = "Did you run all the code?")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1 key:5f8ea1133d
## Exercise 2

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Let's create the names and colors we will use to plot the correlation matrix.  Later, we will also use these colors plot each distillery geographically.  Create a dictionary region_colors with regions as keys and cluster colors as values.

*** =hint
- Use `zip` to combine `regions` and `cluster_colors`, and use `dict()` to convert this to a `dict`.
- Make sure to print your answer!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]

region_colors = ## ENTER CODE HERE! ##


```

*** =solution
```{python}
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]

region_colors = dict(zip(regions, cluster_colors))

print(region_colors)
```

*** =sct
```{python}
test_object("region_colors",
            undefined_msg = "Did you define `region_colors`?",
            incorrect_msg = "It looks like `region_colors` wasn't defined correctly.")
test_function("print",
              not_called_msg = "Make sure print your answer!",
              incorrect_msg = "It looks like what you've printed is incorrect.")               
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:3f6fcf71bc
## Exercise 3

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Let's define a list of `correlation_colors` for distillery pairs that make it clear what the patterns are.  Low correlations will be white, and high correlations will be a distinct color for distilleries from the same group, and gray otherwise.  `correlations` is a matrix with rows and columns corresponding to distilleries, and matrix values coresponding to the correlation of the row/column pair.  Edit the code to define `correlation_colors` for each distillery pair to have input `'white'` then their correlation is less than 0.7.
- `whisky.Group` is a `pandas` dataframe column consisting of distillery group memberships.  For distillery pairs with correlation greater than 0.7, if they share the same whisky group, use the corresponding color from `cluster_colors`.
- Otherwise, define the `correlation_colors` value for that distillery pair as `'lightgrey'`.

*** =hint
- You can index the `(i,j)` distillery pair of `correlations` using `correlations[i,j]`.  How can you test if this value is less than 0.7?
- You can find the group membership of distillery `i` or `j` using `whisky.Group[i]` or `whisky.Group[j]`.  How can you test for their equality?

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
region_colors = dict(zip(regions, cluster_colors))
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv(data_filepath + "whiskies.txt")
whisky["Region"] = pd.read_csv(data_filepath + "regions.txt")
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
distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if ## ENTER CODE HERE! ##                      # if low correlation,
            correlation_colors.append('white')         # just use white.
        else:                                          # otherwise,
            if ## ENTER CODE HERE! ##                  # if the groups match,
                correlation_colors.append(cluster_colors[whisky.Group[i]]) # color them by their mutual group.
            else:                                      # otherwise
                correlation_colors.append('lightgrey') # color them gray.
```

*** =solution
```{python}
distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if correlations[i,j] < .70:                    # if low correlation,
            correlation_colors.append('white')         # just use white.
        else:                                          # otherwise,
            if whisky.Group[i] == whisky.Group[j]:     # if the groups match,
                correlation_colors.append(cluster_colors[whisky.Group[i]]) # color them by their mutual group.
            else:                                      # otherwise
                correlation_colors.append('lightgrey') # color them gray.
```

*** =sct
```{python}
test_object("correlation_colors",
            undefined_msg = "Did you define `correlation_colors`?",
            incorrect_msg = "It looks like `correlation_colors` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:a95b5fe144
## Exercise 4

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Fill in the appropriate code to plot a grid of the distillery correlations.  Color each rectangle in the grid according to `correlation_colors`, and use the correlations themselves as alpha (transparency) values.  Also, when the cursor hovers over a rectangle, output the distillery pair, print both distilleries, as well as the correlation.  Note that `distilleries` contains the distillery names, and `correlations` contains the matrix of distillery correlations by flavor.

*** =hint
- For `"colors"`, the `correlation_colors` we defined in the last question will work as is.
- To convert a numpy matrix (such as `correlations`) to a list, use the `flatten` method.  Try this for both `"alphas"` and `"correlations"`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv(data_filepath + "whiskies.txt")
whisky["Region"] = pd.read_csv(data_filepath + "regions.txt")
model = SpectralCoclustering(n_clusters=6, random_state=0)
flavors = whisky.iloc[:,2:14] # extract flavor attributes
corr_whisky = pd.DataFrame.corr(flavors.transpose())
model.fit(corr_whisky)
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
region_colors = dict(zip(regions, cluster_colors))
distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if correlations[i,j] < .70:
            correlation_colors.append('white')
        else:
            if whisky.Group[i] == whisky.Group[j]:
                correlation_colors.append(cluster_colors[whisky.Group[i]])
            else:
                correlation_colors.append('lightgrey')
```

*** =sample_code
```{python}
source = ColumnDataSource(
    data = {
        "x": np.repeat(distilleries,len(distilleries)),
        "y": list(distilleries)*len(distilleries),
        "colors": ## ENTER CODE HERE! ##,
        "alphas": ## ENTER CODE HERE! ##,
        "correlations": ## ENTER CODE HERE! ##,
    }
)

output_file("Whisky Correlations.html", title="Whisky Correlations")
fig = figure(title="Whisky Correlations",
    x_axis_location="above", tools="resize,hover,save",
    x_range=list(reversed(distilleries)), y_range=distilleries)
fig.plot_width  = 400
fig.plot_height = 400
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.axis.major_label_standoff = 0
fig.xaxis.major_label_orientation = np.pi/3
fig.title_text_font_size="1"

fig.rect('x', 'y', .9, .9, source=source, #1
     color='colors', alpha='alphas')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y", #2
    "Correlation": "@correlations", #3
}
show(fig)
```

*** =solution
```{python}
source = ColumnDataSource(
    data = {
        "x": np.repeat(distilleries,len(distilleries)),
        "y": list(distilleries)*len(distilleries),
        "colors": correlation_colors,
        "alphas": correlations.flatten(),
        "correlations": correlations.flatten(),
    }
)

output_file("Whisky Correlations.html", title="Whisky Correlations")
fig = figure(title="Whisky Correlations",
    x_axis_location="above", tools="resize,hover,save",
    x_range=list(reversed(distilleries)), y_range=distilleries)
fig.plot_width  = 400
fig.plot_height = 400
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.axis.major_label_standoff = 0
fig.xaxis.major_label_orientation = np.pi/3
fig.title_text_font_size="1"

fig.rect('x', 'y', .9, .9, source=source, #1
     color='colors', alpha='alphas')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y", #2
    "Correlation": "@correlations", #3
}
show(fig)
```

*** =sct
```{python}
test_object("source",
            undefined_msg = "Did you define `source`?",
            incorrect_msg = "It looks like `source` wasn't defined correctly.")
test_student_typed("show(fig)",
              pattern=False,
              not_typed_msg="Did you make sure to plot the figure?")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:8c4771d390
## Exercise 5

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Next, we will plot the an example of plotting geographical points. Run the following code, to be adapted in the next section.  Compare this code to that used in plotting the distillery correlations.

*** =hint
- Just run the code and follow along!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
```

*** =sample_code
```{python}
points = [(0,0), (1,2), (3,1)]
xs, ys = zip(*points)
colors = ["red", "blue", "green"]

output_file("Spatial_Example.html", title="Regional Example")
location_source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
    }
)

fig = figure(title = "Title",
    x_axis_location = "above", tools="resize, hover, save")
fig.plot_width  = 300
fig.plot_height = 380
fig.circle("x", "y", 10, 10, size=10, source=location_source,
     color='colors', line_color = None)

hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Location": "(@x, @y)"
}
show(fig)
```

*** =solution
```{python}
points = [(0,0), (1,2), (3,1)]
xs, ys = zip(*points)
colors = ["red", "blue", "green"]

output_file("Spatial_Example.html", title="Regional Example")
location_source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
    }
)

fig = figure(title = "Title",
    x_axis_location = "above", tools="resize, hover, save")
fig.plot_width  = 300
fig.plot_height = 380
fig.circle("x", "y", 10, 10, size=10, source=location_source,
     color='colors', line_color = None)

hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Location": "(@x, @y)"
}
show(fig)
```

*** =sct
```{python}
test_student_typed("show(fig)",
              pattern=False,
              not_typed_msg="Did you make sure to plot the figure?")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:fd58851485
## Exercise 6

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Adapt the given code to define a function `location_plot(title, colors)`.  This function takes a string title and a list of colors corresponding to each distillery, and outputs a Bokeh plot of each distillery by latitude and longitude, and includes text of distillery, latitude, and longitude as the cursor hovers over each point.
- `whisky.Region` is a `pandas` column containing the regional group membership for each distillery.  Use a list comprehension to make a list of the value of `region_colors` for each distillery, and store this list as `region_cols`.
- `location_plot` is still defined form the last exercise.  Use it to plot each distillery, colored by its regional grouping.

*** =hint
-  Recall that the function needs to be defined by adding `def`.
- You can iterate through `whisky.Region` by casting it as a `list`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv(data_filepath + "whiskies.txt")
whisky["Region"] = pd.read_csv(data_filepath + "regions.txt")
model = SpectralCoclustering(n_clusters=6, random_state=0)
flavors = whisky.iloc[:,2:14] # extract flavor attributes
corr_whisky = pd.DataFrame.corr(flavors.transpose())
model.fit(corr_whisky)
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
region_colors = dict(zip(regions, cluster_colors))
```

*** =sample_code
```{python}
# edit this to make the function `location_plot`.

output_file(title+".html")
location_source = ColumnDataSource(
    data={
        "x": whisky[" Latitude"],
        "y": whisky[" Longitude"],
        "colors": colors,
        "regions": whisky.Region,
        "distilleries": whisky.Distillery
    }
)

fig = figure(title = title, #1
    x_axis_location = "above", tools="resize, hover, save")
fig.plot_width  = 400
fig.plot_height = 500
fig.circle("x", "y", 10, 10, size=9, source=location_source,
     color='colors', line_color = None) #2
fig.xaxis.major_label_orientation = np.pi/3
hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Distillery": "@distilleries", #3
    "Location": "(@x, @y)" #4
}
show(fig)

region_cols = ## ENTER CODE HERE! ##
location_plot("Whisky Locations and Regions", region_cols)


```

*** =solution
```{python}
def location_plot(title, colors):
    output_file(title+".html")
    location_source = ColumnDataSource(
        data={
            "x": whisky[" Latitude"],
            "y": whisky[" Longitude"],
            "colors": colors,
            "regions": whisky.Region,
            "distilleries": whisky.Distillery
        }
    )

    fig = figure(title = title, #1
        x_axis_location = "above", tools="resize, hover, save")
    fig.plot_width  = 400
    fig.plot_height = 500
    fig.circle("x", "y", 10, 10, size=9, source=location_source,
         color='colors', line_color = None) #2
    fig.xaxis.major_label_orientation = np.pi/3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries", #3
        "Location": "(@x, @y)" #4
    }
    show(fig)
    
region_cols = [region_colors[i] for i in list(whisky["Region"])]
location_plot("Whisky Locations and Regions", region_cols)    
```

*** =sct
```{python}
test_student_typed("def location_plot(",
              pattern=False,
              not_typed_msg="Did you define `location_plot`?")
test_object("region_cols",
            undefined_msg = "Did you define `region_cols`?",
            incorrect_msg = "It looks like `region_cols` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:2952d41efd
## Exercise 7

In these exercises, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Use list comprehensions to find the the `region_cols` color corresponding to each whisky in `whisky.Region`.  Similarly, find the `cluster_cols` color corresponding to each cluster membership in `whisky.Group`.
- `location_plot` is still loaded from the previous exercise.  Use it to create two interactive plots of distilleries, one colored by defined region called `region_cols`, and one with colors defined by coclustering called `classification_cols`.

*** =hint
- This problem asks you to repeat part of the previous problem (for comparison), and to define a similar color classification for flavor clusters.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
from sklearn.cluster.bicluster import SpectralCoclustering
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np, pandas as pd
whisky = pd.read_csv(data_filepath + "whiskies.txt")
whisky["Region"] = pd.read_csv(data_filepath + "regions.txt")
model = SpectralCoclustering(n_clusters=6, random_state=0)
flavors = whisky.iloc[:,2:14] # extract flavor attributes
corr_whisky = pd.DataFrame.corr(flavors.transpose())
model.fit(corr_whisky)
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
region_colors = dict(zip(regions, cluster_colors))
def location_plot(title, colors):
    output_file(title+".html")
    location_source = ColumnDataSource(
        data={
            "x": whisky[" Latitude"],
            "y": whisky[" Longitude"],
            "colors": colors,
            "regions": whisky.Region,
            "distilleries": whisky.Distillery
        }
    )

    fig = figure(title = title, #1
        x_axis_location = "above", tools="resize, hover, save")
    fig.plot_height = 500
    fig.plot_width = 400
    fig.circle("x", "y", 10, 10, size=9, source=location_source,
         color='colors', line_color = None) #2
    fig.xaxis.major_label_orientation = np.pi/3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries", #3
        "Location": "(@x, @y)" #4
    }
    show(fig)
```

*** =sample_code
```{python}
region_cols = ## ENTER CODE HERE! ##
classification_cols = ## ENTER CODE HERE! ##

location_plot("Whisky Locations and Regions", region_cols)
location_plot("Whisky Locations and Groups", classification_cols)
```

*** =solution
```{python}
region_cols = [region_colors[i] for i in list(whisky.Region)]
classification_cols = [cluster_colors[i] for i in list(whisky.Group)]

location_plot("Whisky Locations and Regions", region_cols)
location_plot("Whisky Locations and Groups", classification_cols)
```

*** =sct
```{python}
test_object("region_cols",
            undefined_msg = "Did you define `region_cols`?",
            incorrect_msg = "It looks like `region_cols` wasn't defined correctly.")
test_object("classification_cols",
            undefined_msg = "Did you define `classification_cols`?",
            incorrect_msg = "It looks like `classification_cols` wasn't defined correctly.")
test_student_typed("location_plot",
              pattern=False,
              not_typed_msg="Did you make sure to use `location_plot` to see your results?")         
success_msg("Great work!")
```

