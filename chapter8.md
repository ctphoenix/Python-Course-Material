---
title       : Module 6 (Bird Watching) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `pandas` makes it easy to perform basic operations on groups within a dataframe without needing to loop through the dataframe. The sample code shows you how to group the dataframe by `birdname` and then find the average `speed_2d` for each bird. Modify the code to assign the maximum altitudes of each bird into an object called `max_altitudes`.

*** =hint
- When grouping by more than one column, remember to use a `[list]`.
- See `?pd.DataFrame.groupby` for help.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
birddata = pd.read_csv("bird_tracking.csv")
```

*** =sample_code
```{python}
## First, use `groupby` to group up the data
grouped_birds = birddata.groupby('bird_name')

## Now operations are performed on each group
mean_speeds = grouped_birds.speed_2d.mean()

## This works for **all** functions. For example, when
## we use `head`, it takes the first 5 lines of each bird
grouped_birds.head()

## Find the maximum `altitude` for each bird.
## Remember to assign this to `max_altitudes`
max_altitudes = ## YOUR CODE HERE ##
```

*** =solution
```{python}
## First, use `groupby` to group up the data
grouped_birds = birddata.groupby('bird_name')

## Now operations are performed on each group
mean_speeds = grouped_birds.speed_2d.mean()

## This works for **all** functions. For example, when
## we use `head`, it takes the first 5 lines of each bird
grouped_birds.head()

## Find the maximum `altitude` for each bird.
## Remember to assign this to `max_altitudes`
max_altitudes = grouped_birds.altitude.max()
```

*** =sct
```{python}
#test_function("",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("grouped_birds",
            undefined_msg = "Did you define `grouped_birds`?",
            incorrect_msg = "It looks like `grouped_birds` wasn't defined correctly.")
test_object("mean_speeds",
            undefined_msg = "Did you define `mean_speeds`?",
            incorrect_msg = "It looks like `mean_speeds` wasn't defined correctly.")
test_object("max_altitudes",
            undefined_msg = "Did you define `max_altitudes`?",
            incorrect_msg = "It looks like `max_altitudes` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1 key:88166cd5b1
## Exercise 2

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `pandas` was designed for time series (i.e., **pan**el **da**ta) and has a useful function called `dt.normalize()` which (among other things) can be used to collapse timestamped data into days. Here’s an example of normalizing our `timestamp` into days.
-  Now `groupby` the data and calculate the maximum altitude per day. Save these results into an object called `max_altitudes_perday`.

*** =hint
- When grouping by more than one column, remember to use a `[list]`.
- See `?pd.DataFrame.groupby` for help.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
birddata = pd.read_csv("bird_tracking.csv")
```

*** =sample_code
```{python}
## Create a new column of day of observation
birddata['n_time'] = birddata.timestamp.dt.normalize()
birddata.n_time.head()

## YOUR CODE HERE ##
```

*** =solution
```{python}
## 
birddata.date_time = pd.to_datetime(birddata.date_time)

## Create a new column of day of observation
birddata['n_time'] = birddata.date_time.dt.date
birddata.n_time.head()

grouped_bydates = birddata.groupby('n_time')
max_altitudes_perday = grouped_bydates.altitude.max()
```

*** =sct
```{python}
#test_function("",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("grouped_bydates",
            undefined_msg = "Did you define `grouped_bydates`?",
            incorrect_msg = "It looks like `grouped_bydates` wasn't defined correctly.")
test_object("max_altitudes_perday",
            undefined_msg = "Did you define `max_altitudes_perday`?",
            incorrect_msg = "It looks like `max_altitudes_perday` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:d421739915
## Exercise 3

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  Now, let’s combine the last two tasks to recreate one of the video lessons. We will `groupby` using both `bird_name` and `n_time`. Then we will find the average speed per day, per bird.
-  First, create a new grouped dataframe called `grouped_birdday` that groups the data by both `bird_name` and `n_time`.

*** =hint
- When grouping by more than one column, remember to use a `[list]`.
- See `?pd.DataFrame.groupby` for help.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
birddata = pd.read_csv("bird_tracking.csv")
birddata['n_time'] = birddata.timestamp.dt.normalize()

```

*** =sample_code
```{python}
grouped_birdday = ## YOUR CODE HERE ##
max_altitudes_perday = grouped_birdday.altitude.max()
max_altitudes_perday.head()
```

*** =solution
```{python}
grouped_birdday = birddata.groupby(['bird_name', 'n_time'])
max_altitudes_perday = grouped_birdday.altitude.max()
max_altitudes_perday.head()

```

*** =sct
```{python}
#test_function("",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.")
test_object("df1",
            undefined_msg = "Did you define `df1`?",
            incorrect_msg = "It looks like `df1` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:c5edc7b67f
## Exercise 4

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  Great! Now you have a dataframe called `grouped_birdday` that has grouped all of the `birddata` by `bird_name` and `n_time`. Now, we can perform the same operations as before – such as using `.mean()` on `speed_2d` to get the average speed per day per bird.
-  We’ve recreated the `Eric` plot using this method for you. Now create two more dataframes – one for `Sanne` and one for `Nico` – and plot all three speeds on the same plot.

*** =hint
- When grouping by more than one column, remember to use a `[list]`.
- See `?pd.DataFrame.groupby` for help.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
birddata = pd.read_csv("bird_tracking.csv")
birddata['n_time'] = birddata.timestamp.dt.normalize()
```

*** =sample_code
```{python}
eric_daily_speed  = grouped_birdday.speed_2d.mean()['Eric']
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
eric_daily_speed  = grouped_birdday.speed_2d.mean()['Eric']
sanne_daily_speed = grouped_birdday.speed_2d.mean()['Sanne']
nico_daily_speed  = grouped_birdday.speed_2d.mean()['Nico']

eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()

```

*** =sct
```{python}
#test_function("",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("eric_daily_speed",
            undefined_msg = "Did you define `eric_daily_speed`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.")
test_object("sanne_daily_speed",
            undefined_msg = "Did you define `sanne_daily_speed`?",
            incorrect_msg = "It looks like `sanne_daily_speed` wasn't defined correctly.")
test_object("nico_daily_speed",
            undefined_msg = "Did you define `nico_daily_speed`?",
            incorrect_msg = "It looks like `nico_daily_speed` wasn't defined correctly.")
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you make sure to plot the daily speeds?")
            
success_msg("Great work!")
```

