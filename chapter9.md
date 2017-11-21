---
title       : Case Study 7 - Movie Analysis: Data Cleaning
description : [The Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this Case Study, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

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

In this exercise, we will define the regression and classification outcomes. Specifically, we will use the revenue column as the target for regression. For classification, we will use an indicator as to whether each movie was profitable or not. The dataset does not yet contain a column with this information, but determine it from other columns.


*** =instructions

- Create a new column in `df` called `profitable`, defined as 1 if the movie revenue is larger than the movie budget, and 0 otherwise.
- Let's define and store the outcomes we will use for regression and classification.
- Define `regression_target` as `'revenue"`, and `classification_target` as `'profitable'`.


*** =hint

- To create `df['profitable']`, use a simple inequality between the budget and revenue columns in `df`.  Then, we will cast this to an integer (1 if true, and 0 otherwise).


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

df['profitable'] = df.budget < df.revenue
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

- Use `df.replace()` to replace any cells with of type `np.inf` or `-np.inf` with `np.nan`.
- Drop all rows with any `np.nan` values in that row using `df.dropna()`. Do any further arguments need to be specified in this function to remove rows with any such values?

*** =hint
- To specify the removal of rows with any missing values, add the parameter `how="any"`

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

Some variables in the dataset are already numeric and perhaps useful for regression and classification. In this exercise, we will store the names of these variables for future use and visualize the data for outcomes and continuous covariates. We will also take a look at the continuous variables by plotting each pair in a scatter matrix, and evaluate the skew of each variable.

*** =instructions
- Call `plt.show()` to observe the plot below. 
    - Consider: Are any continuous covariates related to each other, the outcome, or both?
- Call `skew()` on the columns `outcomes_and_continuous_covariates` in `df`.
    - Consider: Is the skew above 1 for any of these variables?

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

axes = pd.tools.plotting.scatter_matrix(df[outcomes_and_continuous_covariates], alpha = 0.15,color=(0,0,0),hist_kwds={"color":(0,0,0)},facecolor=(1,0,0))
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
success_msg("There is quite a bit of covariance in these pairwise plots, so our modeling strategies or regression and classification might work!")
```


--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:f9c66ebd99
## Exercise 6

It appears that the variables `budget`, `popularity`, `vote_count`, and `revenue` are all right-skewed. In this exercise, transform these covariates to eliminate this skew. Specifically, we will use the `log` transform. Because some of these variables contain values of 0, we must first add a small value (1) to each value to ensure it is strictly positive. (Note that log(0) is negative infinity!)

*** =instructions

- Transform each of the above-mentioned covariates in `df` by `log(1+x)`.

*** =hint
- You can use the `apply()` function on a `df.Series` object. Apply takes a single function as its argument, and returns the `df.Series` with that function applied to each element.
- Functions can be specified anonymously using `lambda` function.

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
for covariate in ['budget','popularity','vote_count','revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log(1+x))
```

*** =sct
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "Did you transform ") 
test_student_typed("df.head()",
              pattern=False,
              not_typed_msg="Did you call `df.head()`?")            
success_msg("Great work!")
```
--- type:NormalExercise lang:python xp:100 skills:2 key:bc061bf4aa
## Exercise 7

In this exercise, we will fit our regression models to predict movie revenue. We will also print the cross-validated correlation between the predicted values and true revenue, and determine the more important variables from the random forests regression fit.

*** =instructions
- Call an instance of `LinearRegression()`, and store as `linear_regression`.
- Call an instance of `RandomForestRegressor()` with `max_depth=4` and, `random_state=0`, and store as `forest_regression`.
- Using both classifiers, call `cross_val_predict()` to fit both classifiers using `df[all_covariates]` and `regression_outcome` with 10 cross-validation folds.
    - Store the predictions as `linear_regression_predicted` and `forest_regression_predicted`, respectively.
- Call `pearsonr()` to compare the accuracy of `regression_outcome` and your cross-validated predictions. How well do these perform?
- Code is provided below to inspect which variables appear to be the most important for predicting profitability according to the random forest model. Which variables are most important?

*** =hint
- This exercise makes heavy use of `sklearn` functions. Feel free to consult its manuals for help.

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
regression_outcome = df[regression_target]

linear_regression =
linear_regression_predicted = 
# determine the correlation of linear regression predictions.

forest_regression =
forest_regression_predicted =
# determine the correlation of random forest predictions.

### Determine feature importance. This code is complete!
forest_regression.fit(df[all_covariates], regression_outcome)
for row in zip(all_covariates, forest_regression.feature_importances_,):
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
for row in zip(all_covariates, forest_regression.feature_importances_,):
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
            not_typed_msg="Did you determine the correlation of `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The cross-validated correlation between the predictions and the outcome is 0.71. Not bad! The cross-validated correlation between the predictions and the outcome is 0.70. Also good, but this fit performs slightly less well than logistic regression.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:56e0886a08
## Exercise 8

In this exercise, we will fit a classification model to determine whether a movie will be profitable or not.
NOTE: let's have them tune max_depth on their own.

In this exercise, we will use both classifiers to determine whether a movie will be profitable or not.

*** =instructions
- Call an instance of `LogisticRegression()`, and store as `linear_classifier`.
- Call an instance of `RandomForestClassifier()` with `max_depth=3` and, `random_state=0`, and store as `forest_classifier`.
- Using both classifiers, call `cross_val_predict()` to fit both classifiers using `df[all_covariates]` and `classification_outcome` with 10 cross-validation folds.
    - Store the predictions as `linear_regression_predicted` and `forest_regression_predicted`, respectively.
- Call `accuracy_score()` to compare the accuracy of `classification_outcome` and your cross-validated predictions. How well do these perform?
- Code is provided below to inspect which variables appear to be the most important for predicting profitability according to the random forest model. Which variables are most important?

*** =hint
- This exercise makes heavy use of `sklearn` functions. Feel free to consult its manuals for help.


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
classification_outcome = df[classification_target]

linear_classifier =
linear_classification_predicted =
# determine the accuracy of logistic regression predictions.


forest_classifier =
forest_classification_predicted =
# determine the accuracy of random forest predictions.

### Determine feature importance. This code is complete!
forest_classifier.fit(df[all_covariates], classification_outcome)
for row in zip(all_covariates, forest_classifier.feature_importances_,):
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
for row in zip(all_covariates, forest_classifier.feature_importances_,):
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
              not_typed_msg="Did you determine the accuracy of  `linear_classifier` and `forest_classifier`?")
success_msg("Great work! The logistic model classifies profitability correctly 82% of the time. The random forests model classifies profitability correctly 80% of the time, slightly less well than the logistic model. We see that according to random forests, popularity and vote count are the most important variables in predicting whether a movie will be profitable.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:dbcd7e671f
## Exercise 9

Finally, let's take a look at the relationship between the predicted revenue and the true revenues. In this exercise, we will visualize the quality of the model fits.

*** =instructions
-  Plot the revenue for each movie again the fits of the linear regression and random forest regression models.
-  Consider: which of the two exhibits a better fit?

*** =hint
- No hint for this one: don't overthink it!

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
success_msg("Great work! It's well worth noting that many movies make zero dollars, which is quite extreme and apparently difficult to predict. Let's see is the random forest model fares any better. Like the linear regression model, predicting whether a movie will make no money at all seem quite difficult.")
```
















