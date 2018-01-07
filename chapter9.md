---
title       : Case Study 7 - Movie Analysis, Part 1 - Data Preparation
description : The [Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this case study, we will use this dataset to determine whether any information about a movie can predict the total revenue of a movie. We will also attempt to predict whether a movie's revenue will exceed its budget. In Part 1, we will inspect, clean, and transform the data. In Part 2, we will use this prepared dataset for analysis. In this exercise, we will import our necessary libraries and read in the dataset.

--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

*** =instructions
- First, we will import several libraries. **sci-kit learn** (`sklearn`) contains helpful statistical models for fitting, and we'll use the `matplotlib.pyplot` library for visualizations. Of course, we will use `numpy`, `scipy`, and `pandas` for data manipulation throughout.
- Read

*** =hint
-  You don't need to do anything, just take a look at the imports.

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
plt.rcParams["figure.figsize"] = (10,10)

```

*** =sample_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)

# Enter code here.


```

*** =solution
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)

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

In this exercise, we will define the regression and classification outcomes. Specifically, we will use the revenue column as the target for regression. For classification, we will construct an indicator of profitability for each movie.


*** =instructions

- Create a new column in `df` called `profitable`, defined as 1 if the movie revenue is greater than the movie budget, and 0 otherwise.
- Next, define and store the outcomes we will use for regression and classification.
    - Define `regression_target` as `'revenue'`.
    - Define `classification_target` as `'profitable'`.


*** =hint

- To create `df['profitable']`, use a simple inequality between the budget and revenue columns in `df`.  Then, we will cast this as an `int`: 1 if true, and 0 otherwise.


*** =pre_exercise_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)
df = pd.read_csv('./merged_movie_data.csv')
```

*** =sample_code
```{python}
# Enter code here.






```

*** =solution
```{python}
regression_target = 'revenue'

df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)

classification_target = 'profitable'

```

*** =sct
```{python}
test_object("df",
            undefined_msg = "Did you define `profitable`?",
            incorrect_msg = "It looks like the column `profitable` wasn't defined correctly.") 
test_object("regression_target",
            undefined_msg = "Did you define `regression_target`?",
            incorrect_msg = "It looks like `regression_target` wasn't defined correctly.") 
test_object("classification_target",
            undefined_msg = "Did you define `classification_target`?",
            incorrect_msg = "It looks like `classification_target` wasn't defined correctly.") 
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:177b5ae318
## Exercise 3

For simplicity, we will proceed by analyzing only the rows without any missing data. In this exercise, we will remove rows with any infinite or missing values.


*** =instructions

- Use `df.replace()` to replace any cells with type `np.inf` or `-np.inf` with `np.nan`.
- Drop all rows with any `np.nan` values in that row using `df.dropna()`. Do any further arguments need to be specified in this function to remove rows with any such values?

*** =hint
- To specify the removal of rows with any missing values, add the parameter `how="any"`.

*** =pre_exercise_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)
df = pd.read_csv('./merged_movie_data.csv')
regression_target = 'revenue'
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)
classification_target = 'profitable'
```

*** =sample_code
```{python}
# Enter code here.




```

*** =solution
```{python}
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")



```

*** =sct
```{python}
test_object("df",
            incorrect_msg = "It looks like not all `np.inf` and `np.nan` cells have been dropped.") 
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:12a3d786b3
## Exercise 4

Many of the variables in our dataframe contain the names of genre, actors/actresses, and keywords. Let's add indicator columns for each genre.

*** =instructions

- Determine all the genres in the genre column. Make sure to use the `strip()` function on each genre to remove trailing characters.
- Next, include each listed genre as a new column in the dataframe. Each element of these genre columns should be 1 if the movie falls under that particular genre, and 0 otherwise.
- Call `df[genres].head()` to view your results.

*** =hint
- No hint for this one.

*** =pre_exercise_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)
df = pd.read_csv('./merged_movie_data.csv')
regression_target = 'revenue'
df['profitable'] = df.budget < df.revenue
df['profitable'] = df['profitable'].astype(int)
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
```

*** =sample_code
```{python}
# Enter your code here.

```

*** =solution
```{python}
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)

df[genres].head()
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

--- type:NormalExercise lang:python xp:100 skills:2 key:9f0ce8e050
## Exercise 5

Some variables in the dataset are already numeric and perhaps useful for regression and classification. In this exercise, we will store the names of these variables for future use and visualize the data for outcomes and continuous covariates. We will also take a look at the continuous variables and outcomes by plotting each pair in a scatter plot, and evaluate the skew of each variable.

*** =instructions
- Call `plt.show()` to observe the plot below. 
    - Consider: Are any continuous covariates related to each other, the outcome, or both?
- Call `skew()` on the columns `outcomes_and_continuous_covariates` in `df`.
    - Consider: Is the skew above 1 for any of these variables?

*** =hint
- No hint for this one.

*** =pre_exercise_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)
df = pd.read_csv('./merged_movie_data.csv')
regression_target = 'revenue'
df['profitable'] = df.budget < df.revenue
df['profitable'] = df['profitable'].astype(int)
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)
```

*** =sample_code
```{python}
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]

axes = pd.tools.plotting.scatter_matrix(df[outcomes_and_continuous_covariates], alpha=0.15, color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
plt.tight_layout()
# show the plot.

# determine the skew.
```

*** =solution
```{python}
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]

axes = pd.tools.plotting.scatter_matrix(df[outcomes_and_continuous_covariates], alpha = 0.15,color=(0,0,0),hist_kwds={"color":(0,0,0)},facecolor=(1,0,0))
plt.tight_layout()
plt.show()

df[outcomes_and_continuous_covariates].skew()

```

*** =sct
```{python}
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")    
              
test_student_typed(".skew()",
              pattern=False,
              not_typed_msg="Did you call `.skew()`?")   
success_msg("Great work! There is quite a bit of covariance in these pairwise plots, so our modeling strategies of regression and classification might work!")
```


--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:f9c66ebd99
## Exercise 6

It appears that the variables `budget`, `popularity`, `vote_count`, and `revenue` are all right-skewed. In this exercise, we will transform these variables to eliminate this skewness. Specifically, we will use the `log` transform. Because some of these variable values are exactly 0, we will add a small value to each value to ensure it is defined. (Note that log(0) is negative infinity!)

*** =instructions

- For each above-mentioned variable in `df`, transform value `x` into `log(1+x)`.

*** =hint
- You can use the `apply()` function on a `df.Series` object. `apply()` takes a single function as its argument, and returns the `df.Series` with that function applied to each element.
- Anonymous functions can be specified using `lambda`.

*** =pre_exercise_code
```{python}
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
plt.rcParams["figure.figsize"] = (10,10)
df = pd.read_csv('./merged_movie_data.csv')
regression_target = 'revenue'
df['profitable'] = df.budget < df.revenue
df['profitable'] = df['profitable'].astype(int)
classification_target = 'profitable'
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
list_genres = df.genres.apply(lambda x: x.split(","))
genres = []
for row in list_genres:
    row = [genre.strip() for genre in row]
    for genre in row:
        if genre not in genres:
            genres.append(genre)

for genre in genres:
    df[genre] = df['genres'].str.contains(genre).astype(int)
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]    
```

*** =sample_code
```{python}
# Enter your code here.



```

*** =solution
```{python}
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log(1+x))
    
    
```

*** =sct
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It appears you did not transform the variables correctly.") 
success_msg("Great work! This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")
```

















