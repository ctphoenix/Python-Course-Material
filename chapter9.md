---
title       : Case Study 7 - Movie Analysis
description : Descriptio of the movie analysis.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

[The Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this Case Study, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.

To perform prediction and classification, we will primarily use the two models we recently discussed: generalized linear regression, and random forests. We will use linear regression to predict revenue, and logistic regression to classify whether a movie made a profit. Random forests come equipped with both a regression and classification mode, and we will use both of them to predict revenue and whether or not a movie made a profit.

In this exercise, we will import our necessary libraries and read in the data.

*** =instructions
- First, we will import several libraries. **sci-kit learn** (`sklearn`) contains helpful models for fitting, and we'll use `matplotlib.pyplot` for visualizations. Of course, we will use `numpy`, `scipy`, and `pandas` for data manipulation throughout.

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
plt.rcParams["figure.figsize"] = (10,10) # Just specifies the size of the plot in this Jupyter Notebook.

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

In this exercise, we will define the regression and classification outcomes.

We will use the revenue column as the target for regression.


*** =instructions

- We will use the revenue column as the target for regression.
- For classification, we will use an indicator as to whether each movie was profitable or not. The dataset does not yet contain a column with this information, but determine it from other columns. Let's define a new column profitable to be a 1 if the movie revenue is larger than the movie budget. For pandas.Series objects, we may do this using a simple inequality, which will return a new pandas.Series object. Then, we will cast this to an integer (1 if true, and 0 otherwise).

- Let's define and store the outcomes we will use for regression and classification.

*** =hint

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

```

*** =solution
```{python}
regression_target = 'revenue'

df['profitable'] = df.budget < df.revenue
df['profitable'] = df['profitable'].astype(int)

classification_target = 'profitable'

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:177b5ae318
## Exercise 3

In this exercise, we will remove rows with missing values.


*** =instructions

- For simplicity, we will proceed by analyzing only the rows without any missing data. Let's drop each row with at least some missing values.

*** =hint

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
```

*** =sample_code
```{python}

```

*** =solution
```{python}
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")
```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:12a3d786b3
## Exercise 4

Many of the variables in our dataframe contain the names of genre, actors/actresses, and keywords. Let's add indicator columns for each genre.

*** =instructions

- Let's determine all the genres in the genre column.
- Next, let's include each listed genre as a new column, each of which will be a 1 if the movie falls under a particular genre, and 0 otherwise.
- Let's look at the dataset again, focusing on our new genres.

*** =hint

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

```

--- type:NormalExercise lang:python xp:100 skills:2 key:9f0ce8e050
## Exercise 5

In this exercise, we will visualize the data for outcomes and continuous covariates.


*** =instructions

- Some variables in the dataset are already numeric and perhaps useful for regression and classification. Let's store the names of these for future use.
- Let's take a look at the continuous variables, and see if they are related to either or both of the outcomes, or each other. We can do this by plotting each pair in a scatter matrix.
- There is quite a bit of covariance in these pairwise plots, so our modeling strategies or regression and classification might work!

*** =hint

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

```


--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:f9c66ebd99
## Exercise 6

It appears that the variables budget, popularity, vote_count, and revenue are all right-skewed. In this exercise, we will transform some covariates to eliminate strong skew.

*** =instructions

- Take the log of each of these variables to even them out.

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}
for covariate in ['budget','popularity','vote_count','revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log(1+x))

axes = pd.tools.plotting.scatter_matrix(df[outcomes_and_continuous_covariates], alpha = 0.15,color=(0,0,0),hist_kwds={"color":(0,0,0)},facecolor=(1,0,0))
plt.tight_layout()
plt.show()   

all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates
df[all_columns]
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

*** =sct
```{python}

```
--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 7

In this exercise, we will fit a regression model.


*** =instructions

- We will now fit our regression models to predict movie revenue. From our imports at the top, we can define the regression model objects and fit them using cross_val_predict. We will print the total correlation between the predicted values and true revenue.

*** =hint

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

```

*** =solution
```{python}
linear_regression = LinearRegression()
regression_outcome = df[regression_target]
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, linear_regression_predicted)

forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, forest_regression_predicted)

# Variable Importance
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_,):
    print(row)

```

*** =sct
```{python}

The cross-validated correlation between the predictions and the outcome is 0.71. Not bad!

The cross-validated correlation between the predictions and the outcome is 0.70. Also good, but this fit performs slightly less well than logistic regression.

```

--- type:NormalExercise lang:python xp:100 skills:2 key:56e0886a08
## Exercise 8

In this exercise, we will fit a classification model to determine whether a movie will be profitable or not.
NOTE: let's have them tune max_depth on their own.

*** =instructions

- Next, we will fit two models to predict whether a movie will be profitable or not.
- Let's finish our analysis by inspecting which variables appear to be the most important for predicting profitability according to the random forest model.
- - How well does this do?

*** =hint

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
linear_regression = LinearRegression()
regression_outcome = df[regression_target]
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_,):
    print(row)
```

*** =sample_code
```{python}

```

*** =solution
```{python}
linear_classifier = LogisticRegression()
classification_outcome = df[classification_target]
linear_classification_predicted = cross_val_predict(linear_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)

forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
forest_classification_predicted = cross_val_predict(forest_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)

forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_,):
    print(row)


```

*** =sct
```{python}
The logistic model classifies profitability correctly 82% of the time.
The random forests model classifies profitability correctly 80% of the time, slightly less well than the logistic model.

We see that according to random forests, popularity and vote count are the most important variables in predicting whether a movie will be profitable.

```

--- type:NormalExercise lang:python xp:100 skills:2 key:dbcd7e671f
## Exercise 9

Finally, let's take a look at the relationship between the predicted revenue and the true revenues. In this exercise, we will visualize the quality of the model fits.

*** =instructions

*** =hint

*** =pre_exercise_code
```{python}
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
linear_regression = LinearRegression()
regression_outcome = df[regression_target]
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_,):
    print(row)
linear_classifier = LogisticRegression()
classification_outcome = df[classification_target]
linear_classification_predicted = cross_val_predict(linear_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)
forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
forest_classification_predicted = cross_val_predict(forest_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)
forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_,):
    print(row)    
```

*** =sample_code
```{python}

```

*** =solution
```{python}
fig, ax = plt.subplots()
ax.scatter(regression_outcome, linear_regression_predicted, edgecolors=(.8, .2, 0, .3), facecolors = (.8, .2, 0, .3),s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]
ax.plot(regression_range, regression_range, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(regression_outcome, forest_regression_predicted, edgecolors=(0, .3, .6, 0.3), facecolors = (0, .3, .6, .3),s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]

ax.plot(regression_range, regression_range, 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```

*** =sct
```{python}
It's well worth noting that many movies make zero dollars, which is quite extreme and apparently difficult to predict. Let's see is the random forest model fares any better.
Like the linear regression model, predicting whether a movie will make no money at all seem quite difficult.
```