---
title       : Case Study 7 - Movie Analysis, Part 2 - Modeling
description : The [Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this Case Study, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.

--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 1

In Part 2 of this case study, we will primarily use the two models we recently discussed: linear/logistic regression and random forests to perform prediction and classification. We will use linear regression to predict revenue, and logistic regression to classify whether a movie was profitable.

In this exercise, we will fit our regression models to predict movie revenue. We will also print the cross-validated accuracy between the predicted values and true revenue, and determine the more important variables from the random forests regression fit.

*** =instructions
- Call an instance of `LinearRegression()`, and assign the output to `linear_regression`.
- Call an instance of `RandomForestRegressor()` with `max_depth=4` and, `random_state=0`, and assign the output to `forest_regression`.
- Call `cross_val_predict()` to fit both classifiers using `df[all_covariates]` and `regression_outcome` with 10 cross-validation folds.
    - Store the predictions as `linear_regression_predicted` and `forest_regression_predicted`, respectively.
- Call `pearsonr()` to compare the accuracy of `regression_outcome` and your cross-validated predictions.
    - Consider: how well do these models perform?
- Code is provided below to determine which variables appear to be the most important for predicting profitability according to the random forest model.
    - Consider: which variables are most important?

*** =hint
- This exercise makes heavy use of `sklearn` functions. Feel free to consult its online documentation for help.

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
```

*** =sample_code
```{python}
regression_outcome = df[regression_target]

linear_regression =
linear_regression_predicted = 
# determine the accuracy of linear regression predictions.

forest_regression =
forest_regression_predicted =
# determine the accuracy of random forest predictions.

### Determine feature importance. This code is complete!
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_):
    print(row)
```

*** =solution
```{python}
regression_outcome = df[regression_target]

linear_regression = LinearRegression()
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, linear_regression_predicted)
# determine the correlation of linear regression predictions.

forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, forest_regression_predicted)
# determine the correlation of random forest predictions.

### Determine feature importance. This code is complete!
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_):
    print(row)
```

*** =sct
```{python}
test_object("linear_regression",
            undefined_msg = "Did you define `linear_regression`?",
            incorrect_msg = "It looks like `linear_regression` wasn't defined correctly.") 
test_object("forest_regression",
            undefined_msg = "Did you define `forest_regression`?",
            incorrect_msg = "It looks like `forest_regression` wasn't defined correctly.") 
test_object("linear_regression_predicted",
            undefined_msg = "Did you define `linear_regression_predicted`?",
            incorrect_msg = "It looks like `linear_regression_predicted` wasn't defined correctly.") 
test_object("forest_regression_predicted",
            undefined_msg = "Did you define `forest_regression_predicted"`?",
            incorrect_msg = "It looks like `forest_regression_predicted"` wasn't defined correctly.") 
test_student_typed("pearsonr",
            pattern=False,
            not_typed_msg="Did you determine the accuracy of `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The cross-validated accuracy between the predictions and the outcome is 0.71. Not bad! The cross-validated accuracy between the predictions and the outcome is 0.70. Also good, but this fit performs slightly less well than logistic regression.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:56e0886a08
## Exercise 2

In this exercise, we will use both classifiers to determine whether a movie will be profitable or not.

*** =instructions
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
classification_outcome = df[classification_target]

linear_classifier = LogisticRegression()
linear_classification_predicted = cross_val_predict(linear_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)

forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
forest_classification_predicted = cross_val_predict(forest_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)

forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_):
    print(row)


```

*** =sct
```{python}
test_object("linear_classifier",
            undefined_msg = "Did you define `linear_classifier`?",
            incorrect_msg = "It looks like `linear_classifier` wasn't defined correctly.") 
test_object("forest_classifier",
            undefined_msg = "Did you define `forest_classifier`?",
            incorrect_msg = "It looks like `forest_classifier` wasn't defined correctly.") 
test_object("linear_classification_predicted",
            undefined_msg = "Did you define `linear_classification_predicted`?",
            incorrect_msg = "It looks like `linear_classification_predicted` wasn't defined correctly.") 
test_object("forest_classification_predicted",
            undefined_msg = "Did you define `forest_classification_predicted"`?",
            incorrect_msg = "It looks like `forest_classification_predicted"` wasn't defined correctly.") 
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
fig, ax = plt.subplots()
ax.scatter(regression_outcome, linear_regression_predicted, edgecolors=(.8, .2, 0, .3), facecolors = (.8, .2, 0, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]
ax.plot(regression_range, regression_range, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# Show the plot.

fig, ax = plt.subplots()
ax.scatter(regression_outcome, forest_regression_predicted, edgecolors=(0, .3, .6, 0.3), facecolors=(0, .3, .6, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]

ax.plot(regression_range, regression_range, 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

# Show the plot.
```

*** =solution
```{python}
fig, ax = plt.subplots()
ax.scatter(regression_outcome, linear_regression_predicted, edgecolors=(.8, .2, 0, .3), facecolors=(.8, .2, 0, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]
ax.plot(regression_range, regression_range, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(regression_outcome, forest_regression_predicted, edgecolors=(0, .3, .6, 0.3), facecolors = (0, .3, .6, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]

ax.plot(regression_range, regression_range, 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
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
```

*** =sample_code
```{python}
positive_revenue_df = 


# Rename the data in the following code, and run.
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, linear_regression_predicted)

forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, forest_regression_predicted)
```

*** =solution
```{python}
positive_revenue_df = df[df["revenue"]>0]

linear_regression_predicted = cross_val_predict(linear_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, linear_regression_predicted)

forest_regression_predicted = cross_val_predict(forest_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)
pearsonr(regression_outcome, forest_regression_predicted)
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
positive_revenue_df = df[df["revenue"]>0]
linear_regression_predicted = cross_val_predict(linear_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)
forest_regression_predicted = cross_val_predict(forest_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)
linear_classifier = LogisticRegression()
forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
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
classification_outcome = positive_revenue_df[classification_target]

linear_classification_predicted = cross_val_predict(linear_classifier, positive_revenue_df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)

forest_classification_predicted = cross_val_predict(forest_classifier, positive_revenue_df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)

forest_classifier.fit(positive_revenue_df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_):
    print(row)


```

*** =sct
```{python}
test_object("linear_classification_predicted",
            undefined_msg = "Did you define `linear_classification_predicted`?",
            incorrect_msg = "It looks like `linear_classification_predicted` wasn't defined correctly.") 
test_object("forest_classification_predicted",
            undefined_msg = "Did you define `forest_classification_predicted"`?",
            incorrect_msg = "It looks like `forest_classification_predicted"` wasn't defined correctly.") 
test_student_typed("accuracy_score",
              pattern=False,
              not_typed_msg="Did you determine the accuracy of `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The logistic model classifies profitability correctly 82% of the time. The random forests model classifies profitability correctly 80% of the time, slightly less well than the logistic model. We see that according to random forests, popularity and vote count appear to be the most important variables in predicting whether a movie will be profitable.")
success_msg("Great work! By excluding movies with zero reported revenue, we do see that the accuracy of both models is increased. Linear regression still appears to slightly outperform random forests.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:175d2a15e1
## Exercise 6

Finally, let's again plot predicted revenue against true revenues. 

*** =instructions
-  Plot the revenue for each movie again the fits of the linear regression and random forest regression models.
-  Consider: which of the two exhibits a better fit?

*** =hint
- No hint for this one!

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
linear_regression = LinearRegression()
regression_outcome = df[regression_target]
linear_regression_predicted = cross_val_predict(linear_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_regression_predicted = cross_val_predict(forest_regression, df[all_covariates], regression_outcome, cv=10)
forest_regression.fit(df[all_covariates], regression_outcome)
linear_classifier = LogisticRegression()
classification_outcome = df[classification_target]
linear_classification_predicted = cross_val_predict(linear_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, linear_classification_predicted)
forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
forest_classification_predicted = cross_val_predict(forest_classifier, df[all_covariates], classification_outcome, cv=10)
accuracy_score(classification_outcome, forest_classification_predicted)
forest_classifier.fit(df[all_covariates], classification_outcome)

positive_revenue_df = df[df["revenue"]>0]
linear_regression_predicted = cross_val_predict(linear_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)
forest_regression_predicted = cross_val_predict(forest_regression, positive_revenue_df[all_covariates], regression_outcome, cv=10)

classification_outcome = positive_revenue_df[classification_target]

linear_classifier = LogisticRegression()
linear_classification_predicted = cross_val_predict(linear_classifier, positive_revenue_df[all_covariates], classification_outcome, cv=10)
forest_classifier = RandomForestClassifierRandomForestClassifier(max_depth=3, random_state=0)
forest_classification_predicted = cross_val_predict(forest_classifier, positive_revenue_df[all_covariates], classification_outcome, cv=10)
forest_classifier.fit(positive_revenue_df[all_covariates], classification_outcome)
```

*** =sample_code
```{python}
fig, ax = plt.subplots()
ax.scatter(regression_outcome, linear_regression_predicted, edgecolors=(.8, .2, 0, .3), facecolors = (.8, .2, 0, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]
ax.plot(regression_range, regression_range, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(regression_outcome, forest_regression_predicted, edgecolors=(0, .3, .6, 0.3), facecolors = (0, .3, .6, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]

ax.plot(regression_range, regression_range, 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

# Show the plot using plt.show().
```

*** =solution
```{python}
fig, ax = plt.subplots()
ax.scatter(regression_outcome, linear_regression_predicted, edgecolors=(.8, .2, 0, .3), facecolors = (.8, .2, 0, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]
ax.plot(regression_range, regression_range, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(regression_outcome, forest_regression_predicted, edgecolors=(0, .3, .6, 0.3), facecolors = (0, .3, .6, .3), s=40)
regression_range = [regression_outcome.min(), regression_outcome.max()]

ax.plot(regression_range, regression_range, 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```

*** =sct
```{python}
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you call `plt.show()`?")
success_msg("Great work! it seems that omitting movies that are estimated to have made precisely no money improves the model fit. This concludes the case study. You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")
```















```
