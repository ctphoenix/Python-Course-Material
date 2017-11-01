---
title       : Case Study 7 - Movie Analysis
description : Descriptio of the movie analysis.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

[The Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this notebook, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.

To perform prediction and classification, we will primarily use the two models we recently discussed: generalized linear regression, and random forests. We will use linear regression to predict revenue, and logistic regression to classify whether a movie made a profit. Random forests come equipped with both a regression and classification mode, and we will use both of them to predict revenue and whether or not a movie made a profit.

*** =instructions
- First, let's import several libraries we will use. We will primarily use submodules in **sci-kit learn** (`sklearn`) for model fitting, and `matplotlib.pyplot` for visualizations. Of course, we will use `numpy`, `scipy`, and `pandas` for data manipulation throughout.
-  Now let's read in the dataset.
(NOTE: This dataset has already undergone some json and other preprocessing, per Matt's Jupyter Notebook. This may be included at will!)

*** =hint
-  Did you do `this`?

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10) # Just specifies the size of the plot in this Jupyter Notebook.

```

*** =sample_code
```{python}
df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df.head()
```

*** =solution
```{python}
df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
df.head()
```

*** =sct
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.") 
test_student_typed("df.head()",
              pattern=False,
              not_typed_msg="Did you call `df.head()`?")            
success_msg("Great work!")
```







--- type:NormalExercise lang:python xp:100 skills:2 key:e2c40f651a
## Exercise 2


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:177b5ae318
## Exercise 3


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:12a3d786b3
## Exercise 4


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:9f0ce8e050
## Exercise 5


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 6


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```
