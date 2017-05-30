---
title       : Case Study 3 - Practice with Classification
description : In this case study, we will analyze a dataset consisting of an assortment of wines classified into "high quality" and "low quality", and will use k-Nearest Neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341
## Exercise 1

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Read in the data as a `pandas` dataframe using `pd.read_csv`.  The data can be found at `https://s3.amazonaws.com/demo-datasets/wine.csv`.

*** =hint
- `pd.read_csv` will work directly!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
data = # Enter your code here!



```

*** =solution
```{python}
import pandas as pd
data = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")
```

*** =sct
```{python}
test_object("data",
            undefined_msg = "Did you define `data`?",
            incorrect_msg = "It looks like `data` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:68c6754822
## Exercise 2

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
- The dataset remains stored as `data`.  Two columns in `data` are `is_red` and `color`, which are redundant. Drop `color` from the dataset, and save the new dataset as `numeric_data`.

*** =hint
- Pandas dataframes contain the `drop` method - give that a try!
- To make sure this is applied to the column, you might try including the parameter `axis=1`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = # Enter your code here!



```

*** =solution
```{python}
numeric_data = data.drop("color", axis=1)
```

*** =sct
```{python}
test_object("numeric_data",
            undefined_msg = "Did you define `numeric_data`?",
            incorrect_msg = "It looks like `numeric_data` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:8515d59a47
## Exercise 3

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
- To ensure that each variable contributes equally to the kNN classifier, we need to standardize the data.  First, from each variable in `numeric_data`, subtract its mean.  Second, for each variable in `numeric_data`, divide by its standard deviation.  Store your standardized result as `numeric_data`.
- Principal component analysis is a way to take a linear snapshot of the data from several different angles, with each snapshot ordered by how well it aligns with variation in the data. The `sklearn.decomposition` module contains the `PCA` class, which determines the most informative principal components of the data (a matrix with columns corresponding to the principal components).  Use `pca.fit(numeric_data).transform(numeric_data)` to extract the first two principal components from the data.  Store this as `principal_components`.


*** =hint
- You can find the mean and standard deviation along each column of a dataframe by selecting `axis=0` in `np.mean` and `np.std`, respectively.
- Both the `fit` and `transform` function require `numeric_data` as input.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
```

*** =sample_code
```{python}
numeric_data = # Enter your code here!

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=2)
principal_components = # Enter your code here!


```

*** =solution
```{python}
numeric_data = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, axis=0)

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit(numeric_data).transform(numeric_data)

```

*** =sct
```{python}
test_object("principal_components",
            undefined_msg = "Did you define `principal_components`?",
            incorrect_msg = "It looks like `principal_components` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:1b705ec875
## Exercise 4

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  The first two principal components can be accessed using `principal_components[:,0]` and `principal_components[:,1]`.  Store these as `x` and `y` respectively, and plot the first two principal components.  The high and low quality wines will be colored using red and blue.  How well are the two groups of wines separated by the first two principal components?

*** =hint
- The columns of `principal_components` are already ordered.  How can you index `principal_components` to plot the first two components?  Store these as `x` and `y`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
```

*** =sample_code
```{python}
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = # Enter your code here!
y = # Enter your code here!

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.show()


```

*** =solution
```{python}
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.show()
```

*** =sct
```{python}
test_student_typed("principal_components",
              pattern=False,
              not_typed_msg="Did you use `principal_components` to define `x` and y`?")
test_student_typed("plt.show",
              pattern=False,
              not_typed_msg="Did you use `plt.show`?")
success_msg("Great work!  Yes, these differ significantly.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:5817bdff2e
## Exercise 5

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  We are now ready to fit the wine data to our kNN classifier.  Create a function `accuracy(predictions, outcomes)` that takes two lists of the same size as arguments and returns a single number, which is the percentage of elements that are equal for the two lists.
-  Use `accuracy` to compare the percentage of similar elements in `x = np.array([1,2,3])` and `y = np.array([1,2,4])`.
-  Print your answer.

*** =hint
- The `==` operator will test for element-wise equality for `numpy` arrays (1 if equal, and 0 if not).  You can then use `np.mean` to find the fraction of these elements that are equal!
- Note that `np.mean`, when used as described above, will find the fraction of equal values between the lists, not the percentage!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)        
```

*** =sample_code
```{python}
def accuracy(predictions, outcomes):
    # Enter your code here!



```

*** =solution
```{python}
def accuracy(predictions, outcomes):
    """
    Finds the percent of predictions that equal outcomes.
    """
    return 100*np.mean(predictions == outcomes)

x = np.array([1,2,3])
y = np.array([1,2,4])

print(accuracy(x,y))
```

*** =sct
```{python}
test_object("x",
            undefined_msg = "Did you define `x`?",
            incorrect_msg = "It looks like `x` wasn't defined correctly.")
test_object("y",
            undefined_msg = "Did you define `y`?",
            incorrect_msg = "It looks like `y` wasn't defined correctly.")
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")              
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:533b201c9b
## Exercise 6

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  The dataset remains stored as `data`.  Because most wines in the dataset are classified as low quality, one very simple classification rule is to predict that all wines are of low quality.  Use the `accuracy` function (preloaded into memory as defined in Exercise 5) to calculate how many wines in the dataset are of low quality.  Accomplish this by calling `accuracy` with `0` as the first argument, and `data["high_quality"]` as the second argument.
-  Print your result.


*** =hint
- The `accuracy` function should work just fine with `0` as the first argument!
- Compare `0` with the `high_quality` column in `data`.  How can you do that?

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
def accuracy(predictions, outcomes):
    return 100*np.mean(predictions == outcomes)
```

*** =sample_code
```{python}
# Enter code here!



```

*** =solution
```{python}
print(accuracy(0, data["high_quality"]))
```

*** =sct
```{python}
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")            
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:3ae1696ed5
## Exercise 7

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
- Use `knn.predict(numeric_data)` to predict which wines are high and low quality and store the result as `library_predictions`.
- Use `accuracy` to find the accuracy of your predictions, using `library_predictions` as the first argument and `data["high_quality"]` as the second argument.
- Print your answer.  Is this prediction better than the simple classifier in Exercise 6?


*** =hint
- A `KNeighborsClassifier` object contains a `predict` method --- try that on `numeric_data`!
- Make sure to use the `accuracy` function to compare `library_predictions` and `data["high_quality"]`!


*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
def accuracy(predictions, outcomes):
    return 100*np.mean(predictions == outcomes)
```

*** =sample_code
```{python}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
# Enter your code here!



```

*** =solution
```{python}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
print(accuracy(library_predictions, data["high_quality"]))
```

*** =sct
```{python}
test_object("library_predictions",
            undefined_msg = "Did you define `library_predictions`?",
            incorrect_msg = "It looks like `library_predictions` wasn't defined correctly.")
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")            
success_msg("Great work!  Yes, this is better!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:428c9e7854
## Exercise 8

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Unlike the `scikit-learn` function, our homemade kNN classifier does not take any shortcuts in calculating which neighbors are closest to each observation, so it is likely too slow to carry out on the whole dataset.  To circumvent this, fix the random generator using `random.seed(123)`, and select 10 rows from the dataset using `random.sample(range(n_rows), 10)`.  Store this selection as `selection`.

*** =hint
- Make sure to use a `range` object for sampling!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
def accuracy(predictions, outcomes):
    return 100*np.mean(predictions == outcomes)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
```

*** =sample_code
```{python}
n_rows = data.shape[0]

# Enter your code here.


```

*** =solution
```{python}
n_rows = data.shape[0]

random.seed(123)
selection = random.sample(range(n_rows), 10)
```

*** =sct
```{python}
test_function("random.seed",
              not_called_msg = "Make sure to call `random.seed`!",
              incorrect_msg = "Did you set the seed value to `123`?")
test_function("random.sample",
              not_called_msg = "Make sure to call `random.sample`!",
              incorrect_msg = "Check your definition of `random.sample` again.")
test_object("selection",
            undefined_msg = "Did you define `selection`?",
            incorrect_msg = "It looks like `selection` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:b8395f53bf
## Exercise 9

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  The sample of `10` row indices are stored as `selection` from the previous exercise.  For each predictor `p` in `predictors[selection]`, use `knn_predict(p, predictors[training_indices,:], outcomes, k=5)` to predict the quality of each wine in the prediction set, and store these predictions as a `np.array` called `my_predictions`.  Note that `knn_predict` is already defined as in the Case 3 videos.
-  Using the `accuracy` function, compare these results to the selected rows from the `high_quality` variable in `data` using `my_predictions` as the first argument and `data.high_quality[selection]` as the second argument.  Store these results as `percentage`.
-  Print your answer.



*** =hint
- Use `knn_predict` for each value in `predictors[selection]`, with `predictors`, `outcomes` and `k=5` as additional parameters.
- Use `accuracy` to compare your predictions to `data.high_quality[selection]`.
- Make sure to print your answer!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
def accuracy(predictions, outcomes):
    return 100*np.mean(predictions == outcomes)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
n_rows = data.shape[0]
random.seed(9)
selection = random.sample(range(n_rows), 10)

```

*** =sample_code
```{python}
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = # Enter your code here!
percentage = # Enter your code here!


```

*** =solution
```{python}
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = np.array([knn_predict(p, predictors[training_indices,:], outcomes, 5) for p in predictors[selection]])
percentage = accuracy(my_predictions, data.high_quality[selection])
print(percentage)


```

*** =sct
```{python}
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "It appears the arguments you used in `accuracy` are not correct.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")  
test_object("percentage",
            undefined_msg = "Did you define `percentage`?",
            incorrect_msg = "It looks like `percentage` wasn't defined correctly.")
success_msg("Great work! Our accuracy is comparable to the library's function!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```
