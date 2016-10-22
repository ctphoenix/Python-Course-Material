---
title       : Module 6 (Bird Watching) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `pandas` makes it easy to perform basic operations on groups within a dataframe without needing to loop through each value in the dataframe. The sample code shows you how to group the dataframe by `birdname` and then find the average `speed_2d` for each bird. Modify the code to assign the mean altitudes of each bird into an object called `mean_altitudes`.

*** =hint
- `grouped_birds` contains a column called `altitude`.  Find the mean of this column!
- This can be done by calling the `mean()` method of this column.

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


--- type:NormalExercise lang:python xp:100 skills:1 key:88166cd5b1
## Exercise 2

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `pandas` contains a useful type called `tslib.Timestamp`, which allows you to describe the date with `dt.date`. In this problem, we will group the flight times by date, and calculate the mean altitude within that day.
-  Use `groupby` and calculate the mean altitude per day. Save these results into an object called `mean_altitudes_perday`.

*** =hint
- See `?pd.DataFrame.groupby` for help!

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

grouped_bydates = birddata.groupby("date")
mean_altitudes_perday = grouped_bydates.altitude.mean()
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

--- type:NormalExercise lang:python xp:100 skills:1 key:d421739915
## Exercise 3

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `birddata` already contains the `date` column.  We will `groupby` using both `bird_name` and `date` and find the average speed for each bird and day.
-  First, create a new grouped dataframe called `grouped_birdday` that groups the data by both `bird_name` and `date`.

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

--- type:NormalExercise lang:python xp:100 skills:1 key:98d1b86767
## Exercise 4

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  Great! Now you have a dataframe called `grouped_birdday` that has grouped all of the `birddata` by `bird_name` and `date`. Now we can use `.mean()` on `speed_2d` to get the average speed for each bird and day.
-  We’ve recreated the `Eric` plot using this method for you. Now create two more dataframes – one for `Sanne` and one for `Nico` – and plot all three speeds on the same plot.

*** =hint
- Call `grouped_birdday.speed_2d.mean()` and select `'Sanne'` and `'Nico'`, respectively.

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
eric_daily_speed  = grouped_bydates.speed_2d.mean()["Eric"]
sanne_daily_speed = ## YOUR CODE HERE ##
nico_daily_speed  = ## YOUR CODE HERE ##

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
success_msg("Great work!")
```

