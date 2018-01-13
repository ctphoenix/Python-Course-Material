---
title       : Case Study 7 - Movie Analysis, Part 2 - Modeling
description : The [movie dataset on which this case study is based](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [The Movie Database (TMDb)](https://www.themoviedb.org/?language=en). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this case study, we will use this dataset to determine whether any information about a movie can predict the total revenue of a movie. We will also attempt to predict whether a movie's revenue will exceed its budget. In Part 2, we will use the dataset prepared in Part 1 for an applied analysis.

--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 1

In Part 2 of this case study, we will primarily use the two models we recently discussed: linear/logistic regression and random forests to perform prediction and classification. We will use these methods to predict revenue, and logistic regression to classify whether a movie was profitable.

In this exercise, we will instantiate regression and classification models. Code is provided that prepares the covariates and outcomes we will use for data analysis.


*** =instructions
- In turn, instantiate `LinearRegression()`, `LogisticRegression()`, `RandomForestRegressor()`, and `RandomForestClassifier()` objects, and assign them to `linear_regression`, `logistic_regression`, `forest_regression`, and `forest_classifier`, respectively.
    - For the random forests models, specify `max_depth=4` and `random_state=0`.

*** =hint
- No hint for this one!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates
```

*** =sample_code
```{python}
# Define all covariates and outcomes from `df`.
regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Instantiate all regression models and classifiers.
linear_regression = 
logistic_regression = 
forest_regression = 
forest_classifier = 
```

*** =solution
```{python}
# Define all covariates and outcomes from `df`.
regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Instantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
```

*** =sct
```{python}
test_object("regression_outcome",
            undefined_msg = "Did you define `regression_outcome`?",
            incorrect_msg = "It looks like `regression_outcome` wasn't defined correctly.") 
test_object("classification_outcome",
            undefined_msg = "Did you define `classification_outcome`?",
            incorrect_msg = "It looks like `classification_outcome` wasn't defined correctly.") 
test_object("covariates",
            undefined_msg = "Did you define `covariates`?",
            incorrect_msg = "It looks like `covariates` wasn't defined correctly.") 
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:56e0886a08
## Exercise 2

In this exercise, we will create two functions that compute a model's score. For regression models, we will use correlation as the score. For classification models, we will use accuracy as the score.

*** =instructions
- Define a function called `correlation` with arguments `estimator`, `X`, and `y` that computes the correlation between the outcome `y` and the predictions made from using covariates `X` to fit the model `estimator` to `y`.
    - To obtain predictions, the function should use the `fit` method from `estimator`, and the `predict` method from the fitted object.
    - The function should return the first argument from `r2_score` comparing `predictions` and `y`.
- Define a function called `accuracy` with the same arguments and code, substituting `accuracy_score` for `r2_score`.


*** =hint
- This exercise makes use of `sklearn` functions. Feel free to consult its online documentation for help.


*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
```

*** =sample_code
```{python}
# Enter your code here.
```

*** =solution
```{python}
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
    
```

*** =sct
```{python}
test_student_typed("correlation",
              pattern=False,
              not_typed_msg="Did you define `correlation`?")
test_student_typed("r2_score",
              pattern=False,
              not_typed_msg="It looks like `correlation` wasn't defined correctly.")
test_student_typed("accuracy",
              pattern=False,
              not_typed_msg="Did you define `correlation`?")
test_student_typed("accuracy_score",
              pattern=False,
              not_typed_msg="It looks like `correlation` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:dbcd7e671f
## Exercise 3

In this exercise, we will compute the cross-validated performance for the linear and random forest regression models.

*** =instructions
- In turn, call `cross_val_score` using `linear_regression` and `forest regression` as models. Store the output as `linear_regression_scores` and `forest_regression_scores`, respectively.
    - Set the parameters `cv=10` to use 10 folds for cross-validation and `scoring=correlation` to use our correlation function defined in the previous exercise.
- Plotting code has been provided to compare the performance of the two models. Use `plt.show()` to plot the correlation between actual and predicted revenue for each cross-validation fold using the linear and random forest regression models.
    -  Consider: which of the two models exhibits a better fit?

*** =hint
- To determine the necessary arguments for `cross_val_score`, use `help()`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)

```

*** =sample_code
```{python}
# Determine the cross-validated correlation for linear and random forest models.


# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

# Show the plot.
```

*** =solution
```{python}
# Determine the cross-validated accuracy for linear and random forest models.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

plt.show()
```

*** =sct
```{python}
test_object("linear_regression_scores",
            undefined_msg = "Did you define `linear_regression_scores`?",
            incorrect_msg = "It looks like `linear_regression_scores` wasn't defined correctly.") 
test_object("forest_regression_scores",
            undefined_msg = "Did you define `forest_regression_scores`?",
            incorrect_msg = "It looks like `forest_regression_scores` wasn't defined correctly.") 
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! According to the metric of cross-validated correlation, the random forest model clearly outperforms the linear model.")
``` 

--- type:NormalExercise lang:python xp:100 skills:2 key:a0ae0c80a0
## Exercise 4

In this exercise, we will compute the cross-validated performance for the linear and random forest classification models.

*** =instructions
- In turn, call `cross_val_score` using `logistic_regression` and `forest_classifier` as models. Store the output as `logistic_regression_scores` and `forest_classification_scores`, respectively.
    - Set the parameters `cv=10` to use 10 folds for cross-validation and `scoring=accuracy` to use our correlation function defined in the previous exercise.
- Plotting code has been provided to compare the performance of the two models. Use `plt.show()` to plot the accuracy of predicted profitability for each cross-validation fold using the logistic and random forest classification models.
    -  Consider: which of the two models exhibits a better fit?

*** =hint
- To determine the necessary arguments for `cross_val_score`, use `help()`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

```

*** =sample_code
```{python}
# Determine the cross-validated accuracy for logistic and random forest models.


# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

# Show the plot.

```

*** =solution
```{python}
# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.show()

```

*** =sct
```{python}
test_object("logistic_regression_scores",
            undefined_msg = "Did you define `logistic_regression_scores`?",
            incorrect_msg = "It looks like `logistic_regression_scores` wasn't defined correctly.") 
test_object("forest_classification_scores",
            undefined_msg = "Did you define `forest_classification_scores`?",
            incorrect_msg = "It looks like `forest_classification_scores` wasn't defined correctly.") 
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! According to the metric of cross-validated accuracy, the random forest model clearly outperforms the linear model, although both perform well.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:8203914a10
## Exercise 5
In Exercise 3, we saw that predicting revenue was only moderately successful. It might be the case that predicting movies that generated precisely no revenue is difficult. In the next three exercises, we will exclude these movies, and rerun the analyses to determine if the fits improve. In this exercise, we will rerun the regression analysis for this subsetted dataset.

*** =instructions
- Define `positive_revenue_df` as the subset of movies in `df` with revenue greater than zero.
- Code is provided below that creates new instances of model objects. Replace all instances of `df` with `positive_revenue_df`, and run the given code.

*** =hint
- `pandas` supports slicing syntax for rows. You can use this to select only rows meeting the logical condition `df["revenue"] > 0`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
```

*** =sample_code
```{python}
positive_revenue_df = 

# Replace the dataframe in the following code, and run.

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Reinstantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

```

*** =solution
```{python}

positive_revenue_df = df[df["revenue"] > 0]

regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]

# Reinstantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)
```


*** =sct
```{python}
test_object("positive_revenue_df",
            undefined_msg = "Did you define `positive_revenue_df`?",
            incorrect_msg = "It looks like `positive_revenue_df` wasn't defined correctly.") 
success_msg("Great work!")

```



--- type:NormalExercise lang:python xp:100 skills:2 key:fe450a86a0
## Exercise 6

In this exercise, we will compute the cross-validated performance for the linear and random forest regression models for positive revenue movies only.

*** =instructions
- In turn, call `cross_val_score` using `linear_regression` and `forest regression` as models. Store the output as `linear_regression_scores` and `forest_regression_scores`, respectively.
    - Set the parameters `cv=10` to use 10 folds for cross-validation and `scoring=correlation` to use our correlation function defined in the previous exercise.
- Plotting code has been provided to compare the performance of the two models. Use `plt.show()` to plot the correlation between actual and predicted revenue for each cross-validation fold using the linear and random forest regression models.
    -  Consider: which of the two models exhibits a better fit? Is this result different from what we observed when considering all movies?

*** =hint
- To determine the necessary arguments for `cross_val_score`, use `help()`.


*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

positive_revenue_df = df[df["revenue"] > 0]
regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    
```

*** =sample_code
```{python}
# Determine the cross-validated correlation for linear and random forest models.


# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

# Show the plot.
```

*** =solution
```{python}
# Determine the cross-validated accuracy for linear and random forest models.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

plt.show()
```

*** =sct
```{python}
test_object("linear_regression_scores",
            undefined_msg = "Did you define `linear_regression_scores`?",
            incorrect_msg = "It looks like `linear_regression_scores` wasn't defined correctly.") 
test_object("forest_regression_scores",
            undefined_msg = "Did you define `forest_regression_scores`?",
            incorrect_msg = "It looks like `forest_regression_scores` wasn't defined correctly.") 
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! According to the metric of cross-validated correlation, the random forest model clearly outperforms the linear model for positive revenue movies. This is broadly the same result as what we observed when considering all movies, although these results are significantly better.")
``` 


--- type:NormalExercise lang:python xp:100 skills:2 key:9445151d8f
## Exercise 7

In this exercise, we will compute the cross-validated performance for the linear and random forest classification models for positive revenue movies only.

*** =instructions
- In turn, call `cross_val_score` using `logistic_regression` and `forest_classifier` as models. Store the output as `logistic_regression_scores` and `forest_classification_scores`, respectively.
    - Set the parameters `cv=10` to use 10 folds for cross-validation and `scoring=accuracy` to use our correlation function defined in the previous exercise.
- Plotting code has been provided to compare the performance of the two models. Use `plt.show()` to plot the accuracy of predicted profitability for each cross-validation fold using the logistic and random forest classification models.
    -  Consider: which of the two models exhibits a better fit? Is this result different from what we observed when considering all movies?

*** =hint
- To determine the necessary arguments for `cross_val_score`, use `help()`.


*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

df = pd.read_csv(data_filepath + 'merged_movie_data.csv')
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
for covariate in ['budget', 'popularity', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1+x))    
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

positive_revenue_df = df[df["revenue"] > 0]
regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
```

*** =sample_code
```{python}
# Determine the cross-validated accuracy for logistic and random forest models.


# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

# Show the plot.

```

*** =solution
```{python}
# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.show()

```

*** =sct
```{python}
test_object("logistic_regression_scores",
            undefined_msg = "Did you define `logistic_regression_scores`?",
            incorrect_msg = "It looks like `logistic_regression_scores` wasn't defined correctly.") 
test_object("forest_classification_scores",
            undefined_msg = "Did you define `forest_classification_scores`?",
            incorrect_msg = "It looks like `forest_classification_scores` wasn't defined correctly.") 
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! According to the metric of cross-validated accuracy, the random forest model clearly outperforms the linear model for positive revenue movies. This is broadly the same result as what we observed when considering all movies. This concludes the case study. You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")
```
``` 





