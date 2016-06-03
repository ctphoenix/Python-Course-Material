---
title       : Module 3 (Classification) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
-  Read in the data as a `pandas` dataframe.  The data can be found at `https://s3.amazonaws.com/demo-datasets/wine.csv`.

*** =hint
- 

*** =pre_exercise_code
```{python}
import numpy as np, random, scipy.stats as ss
def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode
def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))
def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]
def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]
```

*** =sample_code
```{python}
import pandas as pd
data = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")
```

*** =solution
```{python}
import pandas as pd
data = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")
```

*** =sct
```{python}
#test_function("print",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("data",
            undefined_msg = "Did you define `data`?",
            incorrect_msg = "It looks like `data` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:68c6754822
## Exercise 2

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
- The dataset contains a variable called "is_red", making "color" redundant. Drop the color variable from the dataset.

*** =hint
- 

*** =pre_exercise_code
```{python}
import numpy as np, random, scipy.stats as ss
def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode
def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))
def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]
def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]
import pandas as pd
data = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")    
```

*** =sample_code
```{python}
numeric_data = data.drop("color", axis=1)
```

*** =solution
```{python}
numeric_data = data.drop("color", axis=1)
```

*** =sct
```{python}
test_function("print",
              not_called_msg = "Make sure to call ``!",
              incorrect_msg = "Check your definition of `` again.")
test_object("",
            undefined_msg = "Did you define ``?",
            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!")
```




