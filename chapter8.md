---
title       : Module 6 (Bird Watching) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these exercises, we will continue taking a look at patterns of bird flights over time.

*** =instructions
-  `pandas` makes it easy to perform basic operations on groups within a dataframe without needing to loop
through the dataframe. The sample code shows you how to group the dataframe by `birdname` and then find the average
`speed_2d` for each bird. Modify the code to assign the maximum altitudes of each bird into an object called
`max_altitudes`.

*** =hint
- 

*** =pre_exercise_code
```{python}
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


  

