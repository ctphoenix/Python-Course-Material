---
title       : Case Study 5 - Bird Migration
description : In this case study, we will continue taking a look at patterns of flight for each of the three birds in our dataset.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341
## Exercise 1

In this case study, we will continue taking a look at patterns of flight for each of the three birds in our dataset.

*** =instructions
- `pandas` makes it easy to perform basic operations on groups within a dataframe without needing to loop through each value in the dataframe. The sample code shows you how to group the dataframe by `birdname` and then find the average `speed_2d` for each bird.  Modify the code to assign the mean altitudes of each bird into an object called `mean_altitudes`.

*** =hint
- `grouped_birds` contains a column called `altitude`.  Find the mean of this column!
- This can be done by calling the `mean()` method on this column.

*** =pre_exercise_code 
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
birddata = pd.read_csv(data_filepath + "bird_tracking.csv")
birddata.date_time = pd.to_datetime(birddata.date_time)
birddata["date"] = birddata.date_time.dt.date
grouped_bydates = birddata.groupby(["bird_name", "date"])

```

*** =sample_code
```{python}
# First, use `groupby` to group up the data.
grouped_birds = birddata.groupby("bird_name")

# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()

# The `head` method prints the first 5 lines of each bird.
grouped_birds.head()

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = ## YOUR CODE HERE ##

```

*** =solution
```{python}
# First, use `groupby` to group up the data.
grouped_birds = birddata.groupby("bird_name")

# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()

# The `head` method prints the first 5 lines of each bird.
grouped_birds.head()

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = grouped_birds.altitude.mean()

```

*** =sct
```{python}
test_object("mean_altitudes",
            undefined_msg = "Did you define `mean_altitudes`?",
            incorrect_msg = "It looks like `mean_altitudes` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:88166cd5b1
## Exercise 2

In this case study, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  In this exercise, we will group the flight times by date and calculate the mean altitude within that day.  Use `groupby` to group the data by date.
-  Calculate the mean altitude per day and store these results as `mean_altitudes_perday`.

*** =hint
- See `?pd.DataFrame.groupby` for help!
- For `mean_altitudes_perday`, you can use the `mean()` method on `grouped_bydates.altitude`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
birddata = pd.read_csv(data_filepath + "bird_tracking.csv")
```

*** =sample_code
```{python}
# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)

# Create a new column of day of observation
birddata["date"] = birddata.date_time.dt.date

# Check the head of the column.
birddata.date.head()

grouped_bydates = ## YOUR CODE HERE ##
mean_altitudes_perday = ## YOUR CODE HERE ##
```

*** =solution
```{python}
# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)

# Create a new column of day of observation
birddata["date"] = birddata.date_time.dt.date

# Check the head of the column.
birddata.date.head()

grouped_bydates = birddata.groupby("date")
mean_altitudes_perday = grouped_bydates.altitude.mean()
```




*** =sct
```{python}
test_object("mean_altitudes_perday",
            undefined_msg = "Did you define `mean_altitudes_perday`?",
            incorrect_msg = "It looks like `mean_altitudes_perday` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:d421739915
## Exercise 3

In this case study, we will continue taking a look at patterns of bird flights over time.

*** =instructions
- `birddata` already contains the `date` column.  To find the average speed for each bird and day, create a new grouped dataframe called `grouped_birdday` that groups the data by both `bird_name` and `date`.

*** =hint
- When grouping by more than one column, remember to use a `list`.
- See `?pd.DataFrame.groupby` for help!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
birddata = pd.read_csv(data_filepath + "bird_tracking.csv")
birddata.date_time = pd.to_datetime(birddata.date_time)
birddata['date'] = birddata.date_time.dt.date
```

*** =sample_code
```{python}
grouped_birdday = ## YOUR CODE HERE ##
mean_altitudes_perday = grouped_birdday.altitude.mean()

# look at the head of `mean_altitudes_perday`.
mean_altitudes_perday.head()

```

*** =solution
```{python}
grouped_birdday = birddata.groupby(["bird_name", "date"])
mean_altitudes_perday = grouped_birdday.altitude.mean()

# look at the head of `mean_altitudes_perday`.
mean_altitudes_perday.head()

```

*** =sct
```{python}
test_object("mean_altitudes_perday",
            undefined_msg = "Did you define `mean_altitudes_perday`?",
            incorrect_msg = "It looks like `mean_altitudes_perday` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:98d1b86767
## Exercise 4

In this case study, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  Great!  Now find the average speed for each bird and day.  Create three `pandas` Series objects – one for each bird.
-  Use the plotting code provided to plot the average speeds for each bird.

*** =hint
- To find the average, consider how you to use `speed_2d` and `mean` for all three birds.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
birddata = pd.read_csv(data_filepath + "bird_tracking.csv")
birddata.date_time = pd.to_datetime(birddata.date_time)
birddata["date"] = birddata.date_time.dt.date
grouped_bydates = birddata.groupby(["bird_name", "date"])
```

*** =sample_code
```{python}
eric_daily_speed  = # Enter your code here.
sanne_daily_speed = # Enter your code here.
nico_daily_speed  = # Enter your code here.

eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()

```

*** =solution
```{python}
eric_daily_speed  = grouped_bydates.speed_2d.mean()["Eric"]
sanne_daily_speed = grouped_bydates.speed_2d.mean()["Sanne"]
nico_daily_speed  = grouped_bydates.speed_2d.mean()["Nico"]

eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()

```

*** =sct
```{python}
test_object("eric_daily_speed",
            undefined_msg = "Did you define `eric_daily_speed`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.")
test_object("sanne_daily_speed",
            undefined_msg = "Did you define `sanne_daily_speed`?",
            incorrect_msg = "It looks like `sanne_daily_speed` wasn't defined correctly.")
test_object("nico_daily_speed",
            undefined_msg = "Did you define `nico_daily_speed`?",
            incorrect_msg = "It looks like `nico_daily_speed` wasn't defined correctly.")
test_student_typed("plt.show",
              pattern=False,
              not_typed_msg="Did you use `plt.show`?")       
success_msg("Great work!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```

