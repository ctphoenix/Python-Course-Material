---
title       : Case Study 7 - Movie Analysis, Part 2 - Modeling
description : The [Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this Case Study, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.

--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 1

In Part 2 of this case study, we will primarily use the two models we recently discussed: linear/logistic regression and random forests to perform prediction and classification. We will use linear regression to predict revenue, and logistic regression to classify whether a movie was profitable.

Recall that `regression_target`, `classification_target`, and `all_covariates` are strings or lists of strings, and each string is a column name in `df`. In this exercise, we will prepare the covariates and outcomes we will use for data analysis defining `regression_outcome`, `classification_outcome`, and `covariates` as selected columns in `df`. We will also instantiate regression and classification models for fitting.


*** =instructions
- In turn, call instances of `LinearRegression()`, `LogisticRegression()`, `RandomForestRegressor()`, `RandomForestClassifier()`, and assign the output to `linear_regression`, `linear_classifier`, `forest_regression`, and `forest_classifier`, respectively.
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
linear_classifier = 
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
linear_classifier = LogisticRegression()
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

fit our regression models to predict movie revenue. We will also print the cross-validated accuracy between the predicted values and true revenue, and determine the more important variables from the random forests regression fit.
In this exercise, we will use both classifiers to determine whether a movie will be profitable or not.

*** =instructions
- Call an instance of `RandomForestRegressor()` with `max_depth=4` and, `random_state=0`, and assign the output to `forest_regression`.
- Call `cross_val_predict()` to fit both classifiers using `df[all_covariates]` and `regression_outcome` with 10 cross-validation folds.
    - Store the predictions as `linear_regression_predicted` and `forest_regression_predicted`, respectively.
- Call `pearsonr()` to compare the accuracy of `regression_outcome` and your cross-validated predictions.
    - Consider: how well do these models perform?
- Code is provided below to determine which variables appear to be the most important for predicting profitability according to the random forest model.
    - Consider: which variables are most important?

- Call an instance of `LogisticRegression()`, and store as `linear_classifier`.
- Call an instance of `RandomForestClassifier()` with `max_depth=3` and, `random_state=0`, and assign it to `forest_classifier`.
- Call `cross_val_predict()` to fit both classifiers using `df[all_covariates]` and `classification_outcome` with 10 cross-validation folds.
    - Assign these predictions to `linear_regression_predicted` and `forest_regression_predicted`.
- Call `accuracy_score()` to compare the accuracy of `classification_outcome` to your cross-validated predictions.
    - Consider: how well do these perform?
- We provide code below to inspect which variables appear to be the most important for predicting profitability in the random forest model.
    - Consider: which variables are most important?


*** =hint
- This exercise makes heavy use of `sklearn` functions. Feel free to consult its online documentation for help.


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
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
linear_classifier = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
```

*** =sample_code
```{python}
classification_outcome = df[classification_target]

linear_classifier =
linear_classification_predicted =
# determine the accuracy of logistic regression predictions.


forest_classifier =
forest_classification_predicted =
# determine the accuracy of random forest predictions.

### Determine feature importance. This code is complete!
forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_):
    print(row)
```

*** =solution
```{python}
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(X, y)[0]
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(X, y)[0]
```

*** =sct
```{python}
test_object("linear_classification_predicted",
            undefined_msg = "Did you define `linear_classification_predicted`?",
            incorrect_msg = "It looks like `linear_classification_predicted` wasn't defined correctly.") 
test_object("forest_classification_predicted",
            undefined_msg = "Did you define `forest_classification_predicted`?",
            incorrect_msg = "It looks like `forest_classification_predicted` wasn't defined correctly.") 
test_student_typed("accuracy_score",
              pattern=False,
              not_typed_msg="Did you determine the accuracy of `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The logistic model classifies profitability correctly 82% of the time. The random forests model classifies profitability correctly 80% of the time, slightly less well than the logistic model. We see that according to random forests, popularity and vote count appear to be the most important variables in predicting whether a movie will be profitable.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:dbcd7e671f
## Exercise 3

Finally, let's take a look at the relationship between predicted and true revenue. In this exercise, we will visualize the quality of the model fits.

*** =instructions
-  Plot the revenue for each movie against the fits of the linear regression and random forest regression models.
    -  Consider: which of the two models exhibits a better fit?

*** =hint
- No hint for this one: don't overthink it!

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
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
linear_classifier = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(X, y)[0]
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(X, y)[0]

```

*** =sample_code
```{python}
# determine the correlation of random forest predictions.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv = 10, scoring = correlation)

# Plot Results
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.xlim(0.5,1)
plt.ylim(0.5,1)
plt.xlabel("Linear Regression")
plt.ylabel("Forest Regression")

# Show the plot.
```

*** =solution
```{python}
# determine the correlation of random forest predictions.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv = 10, scoring = correlation)

# Plot Results
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.xlim(0.5,1)
plt.ylim(0.5,1)
plt.xlabel("Linear Regression")
plt.ylabel("Forest Regression")
plt.show()
```

*** =sct
```{python}
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! It's well worth noting that many movies make zero dollars, which is quite extreme and apparently difficult to predict. Let's see if the random forest model fares any better. Like the linear regression model, predicting whether a movie will make no money at all seems quite difficult.")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:a0ae0c80a0
## Exercise 4

It appears that predicting movies that are reported to have made precisely no money is difficult. In the next three exercises, we will exclude these movies, and rerun the analyses to determine if the fit improves. In this exercise, we will rerun the regression analysis for this subsetted dataset.

*** =instructions
- Define `positive_revenue_df` as the subset of movies in `df` with revenue greater than zero.
- The solutions to the previous analyses using `df` are shown below. Replace all instances of `df` with `positive_revenue_df`, and run the given code.
- Consider the following comparisons to the analysis that included movies with zero reported revenue: 
    - Are these cross-validated correlations between predictions and true revenue higher or lower in general?
    - Previously, linear regression outperformed random forests. Has this changed?

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
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
linear_classifier = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(X, y)[0]
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(X, y)[0]
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
```

*** =sample_code
```{python}
positive_revenue_df = 

# Rename the data in the following code, and run.
regression_outcome = df[regression_target]
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, linear_regression_predicted)

forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, forest_regression_predicted)
```

*** =solution
```{python}
# determine the correlation of random forest predictions.
linear_classification_scores = cross_val_score(linear_classifier, covariates, classification_outcome, cv = 10, scoring = correlation)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv = 10, scoring = correlation)

# plot results
plt.scatter(linear_classification_scores, forest_classification_scores)
plt.xlim(0.5,1)
plt.ylim(0.5,1)
plt.xlabel("Linear classification")
plt.ylabel("Forest classification")
plt.show()
```

*** =sct
```{python}
test_object("positive_revenue_df",
            undefined_msg = "Did you define `positive_revenue_df`?",
            incorrect_msg = "It looks like `positive_revenue_df` wasn't defined correctly.")
test_object("linear_regression_predicted",
            undefined_msg = "Did you define `linear_regression_predicted`?",
            incorrect_msg = "It looks like `linear_regression_predicted` wasn't defined correctly.") 
test_object("forest_regression_predicted",
            undefined_msg = "Did you define `forest_regression_predicted`?",
            incorrect_msg = "It looks like `forest_regression_predicted` wasn't defined correctly.") 
test_student_typed("pearsonr",
              pattern=False,
              not_typed_msg="Did you determine the correlation between `linear_classifier` and `forest_classifier` with revenue?")
success_msg("Great work! By excluding movies with zero reported revenue, we do see that the correlation between predictions and outcome is increased. Linear regression still appears to slightly outperform random forests.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:8203914a10
## Exercise 5

In this exercise, we will rerun the classification analysis for the subsetted dataset that includes only movies with positive revenue.

*** =instructions
- Replace all instances of `df` with `positive_revenue_df`, and run the given code.
- Consider the following comparisons to the analysis that included movies with zero reported revenue: 
    - Is the cross-validated accuracy between predictions and true revenue higher or lower in general?
    - Previously, logistic regression outperformed random forest classification. Has this changed?

*** =hint
- No hint for this one.


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
continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]   
all_covariates = continuous_covariates + genres
all_columns = [regression_target, classification_target] + all_covariates

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
linear_classifier = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(X, y)[0]
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(X, y)[0]
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv = 10, scoring = correlation)
linear_classification_scores = cross_val_score(linear_classifier, covariates, classification_outcome, cv = 10, scoring = correlation)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv = 10, scoring = correlation)
```

*** =sample_code
```{python}
# Rename the data in the following code, and run.

classification_outcome = df[classification_target]

linear_classification_predicted = cross_val_predict(linear_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)

forest_classification_predicted = cross_val_predict(forest_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)

forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_):
    print(row)


```

*** =solution
```{python}

positive_revenue_df = df[df["revenue"] > 0]
regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

linear_regression = LinearRegression()
linear_classifier = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
```


*** =sct
```{python}
test_object("linear_classification_predicted",
            undefined_msg = "Did you define `linear_classification_predicted`?",
            incorrect_msg = "It looks like `linear_classification_predicted` wasn't defined correctly.") 
test_object("forest_classification_predicted",
            undefined_msg = "Did you define `forest_classification_predicted`?",
            incorrect_msg = "It looks like `forest_classification_predicted` wasn't defined correctly.") 
test_student_typed("accuracy_score",
              pattern=False,
              not_typed_msg="Did you determine the accuracy of `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The logistic model classifies profitability correctly 82% of the time. The random forests model classifies profitability correctly 83% of the time, which is slightly better than the linear model, aa reversal from our previous accuracy results. We see that according to random forests, popularity and vote count appear to be the most important variables in predicting whether a movie will be profitable.")
success_msg("Great work! By excluding movies with zero reported revenue, we do see that the accuracy of both models is increased. Linear regression still appears to slightly outperform random forests.")
```














success_msg("Great work! it seems that omitting movies that are estimated to have made precisely no money improves prediction of revenues. This concludes the case study. You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")









