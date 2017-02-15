---
title       : Case Study 4 - Visualizing Whisky Classification
description : In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341
## Exercise 1

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
-  Here we provide a basic demonstration of an interactive grid plot using Bokeh.  Execute the following code and follow along with the comments. We will later adapt this code to plot the correlations among distillery flavor profiles as well as plot a geographical map of distilleries colored by region and flavor profile.
-  Make sure to study this code now, as we will edit similar code in the exercises that follow.
-  Once you have plotted the code, hover, click, and drag your cursor on the plot to interact with it.  Additionally, explore the icons in the top-right corner of the plot for more interactive options!

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
# First, we import a tool to allow text to pop up on a plot when the cursor
# hovers over it.  Also, we import a data structure used to store arguments
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
# where 0 is completely transparent.

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
# First, we import a tool to allow text to pop up on a plot when the cursor
# hovers over it.  Also, we import a data structure used to store arguments
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
# where 0 is completely transparent.

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
test_student_typed("show(fig)",
              pattern=False,
              not_typed_msg="Did you make sure to plot the figure?")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:5f8ea1133d
## Exercise 2

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Let's create the names and colors we will use to plot the correlation matrix of whisky flavors.  Later, we will also use these colors to plot each distillery geographically.  Create a dictionary `region_colors` with regions as keys and `cluster_colors` as values.
- Print `region_colors`.

*** =hint
- Use `zip` to combine `regions` and `cluster_colors` and use `dict()` to convert this to a `dict`.
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

--- type:NormalExercise lang:python xp:100 skills:2 key:3f6fcf71bc
## Exercise 3

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- `correlations` is a two-dimensional `np.array` with both rows and columns corresponding to distilleries and elements corresponding to the flavor correlation of each row/column pair.  Let's define a list `correlation_colors`, with `string` values corresponding to colors to be used to plot each distillery pair.  Low correlations among distillery pairs will be white, high correlations will be a distinct group color if the distilleries from the same group, and gray otherwise.  Edit the code to define `correlation_colors` for each distillery pair to have input `'white'` if their correlation is less than 0.7.
- `whisky.Group` is a `pandas` dataframe column consisting of distillery group memberships.  For distillery pairs with correlation greater than 0.7, if they share the same whisky group, use the corresponding color from `cluster_colors`.  Otherwise, the `correlation_colors` value for that distillery pair will be defined as `'lightgray'`.	

*** =hint
- You can index the `(i,j)` distillery pair of `correlations` using `correlations[i,j]`.  How can you test if this value is less than 0.7?
- You can find the group membership of distillery `i` using `whisky.Group[i]`.

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
                correlation_colors.append('lightgray') # color them lightgray.
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
                correlation_colors.append('lightgray') # color them lightgray.
```

*** =sct
```{python}
test_object("correlation_colors",
            undefined_msg = "Did you define `correlation_colors`?",
            incorrect_msg = "It looks like `correlation_colors` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:a95b5fe144
## Exercise 4

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- We will edit the following code to make an interactive grid of the correlations among distillery pairs using `correlation_colors` and `correlations`. `correlation_colors` is a list of each distillery pair.  To convert `correlations` from a `np.array` to a `list`, we will use the `flatten` method.  Define the `color` of each rectangle in the grid using to `correlation_colors`.
- Define the `alpha` (transparency) values using `correlations.flatten()`.
- Define `correlations` and using `correlations.flatten()`.  When the cursor hovers over a rectangle, this will output the distillery pair, show both distilleries as well as their correlation coefficient.


*** =hint
- For `"colors"`, the `correlation_colors` we defined in the last question will work as is.
- Use the `flatten` method for both `"alphas"` and `"correlations"`!

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
                correlation_colors.append('lightgray')
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
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.xaxis.major_label_orientation = np.pi / 3

fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='alphas')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y",
    "Correlation": "@correlations",
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
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.xaxis.major_label_orientation = np.pi / 3

fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='alphas')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y",
    "Correlation": "@correlations",
}
show(fig)
```

*** =sct
```{python}
test_student_typed("show(fig)",
              pattern=False,
              not_typed_msg="Did you make sure to plot the figure?")
test_student_typed("correlation_colors",
              pattern=False,
              not_typed_msg="Did you define `colors` correctly?")              
test_student_typed("flatten",
              pattern=False,
              not_typed_msg="Did you use `flatten` to define `alphas` and `correlations`?")        
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:8c4771d390
## Exercise 5

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Next, we provide an example of plotting points geographically. Run the following code, to be adapted in the next section.  Compare this code to that used in plotting the distillery correlations.

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

--- type:NormalExercise lang:python xp:100 skills:2 key:fd58851485
## Exercise 6

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Adapt the given code from the beginning to `show(fig)` in order to define a function `location_plot(title, colors)`.  This function takes a string `title` and a list of colors corresponding to each distillery and outputs a Bokeh plot of each distillery by latitude and longitude.  As the cursor hovers over each point, it displays the distillery name, latitude, and longitude.
- `whisky.Region` is a `pandas` column containing the regional group membership for each distillery.  Make a list consisting of the value of `region_colors` for each distillery, and store this list as `region_cols`.
- Use `location_plot` to plot each distillery, colored by its regional grouping.	


*** =hint
- Remember to define the function!
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

fig = figure(title = title,
    x_axis_location = "above", tools="resize, hover, save")
fig.plot_width  = 400
fig.plot_height = 500
fig.circle("x", "y", 10, 10, size=9, source=location_source,
     color='colors', line_color = None)
fig.xaxis.major_label_orientation = np.pi / 3
hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Distillery": "@distilleries",
    "Location": "(@x, @y)"
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

    fig = figure(title = title,
        x_axis_location = "above", tools="resize, hover, save")
    fig.plot_width  = 400
    fig.plot_height = 500
    fig.circle("x", "y", 10, 10, size=9, source=location_source,
         color='colors', line_color = None)
    fig.xaxis.major_label_orientation = np.pi / 3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries",
        "Location": "(@x, @y)"
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

--- type:NormalExercise lang:python xp:100 skills:2 key:2952d41efd
## Exercise 7

In this case study, we have prepared step-by-step instructions for you on how to prepare plots in Bokeh, a library designed for simple and interactive plotting.  We will demonstrate Bokeh by continuing the analysis of Scotch whiskies.

*** =instructions
- Use list comprehensions to create the list `region_cols` consisting of the color in `region_colors` that corresponds to each whisky in `whisky.Region`.
- Similarly, create a list `classification_cols` consisting of the color in `cluster_colors` that corresponds to each cluster membership in `whisky.Group`.
- `location_plot` remains stored from the previous exercise.  Use it to create two interactive plots of distilleries, one colored by defined region called `region_cols` and the other with colors defined by coclustering designation called `classification_cols`.  How well do the coclustering groupings match the regional groupings?

*** =hint
- This problem asks you to repeat part of the previous problem (for comparison), and to define a similar color classification for flavor clusters.  Two straightforward list comprehensions will do the trick.
- Consider casting `whisky.Region` and `whisky.Group` as lists!

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

    fig = figure(title = title,
        x_axis_location = "above", tools="resize, hover, save")
    fig.plot_height = 500
    fig.plot_width = 400
    fig.circle("x", "y", 10, 10, size=9, source=location_source,
         color='colors', line_color = None)
    fig.xaxis.major_label_orientation = np.pi / 3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries",
        "Location": "(@x, @y)"
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
success_msg("Great work!  We see that there is not very much overlap between the regional classifications and the coclustering classifications.  This means that regional classifications are not a very good guide to Scotch whisky flavor profiles.  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```





