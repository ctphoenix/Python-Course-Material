---
title       : Homework - Practice with Classification
description : In this homework, we will analyze a dataset consisting of an assortment of wines classified into "high quality" and "low quality", and will use k-Nearest Neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341
## Exercise 1

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Read in the data as a `pandas` dataframe.  The data can be found at `https://s3.amazonaws.com/demo-datasets/wine.csv`.

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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
- To ensure that each variable contributes equally to the kNN classifier, we need to standardize the data.  First, from each variable in `numeric_data`, subtract its mean.  Second, for each variable in `numeric_data`, divide by its standard deviation.  Store this again as `numeric_data`.
- Principal component analysis is a way to take a linear snapshot of the data from several different angles, with each snapshot ordered by how well it aligns with variation in the data. Use the `PCA` function in the scikit-learn (`sklearn`) library to find and store the two most informative principal components of the data (a matrix with two columns corresponding to the principal components), and store it as `pca`.
- Use the `fit` and `transform` methods on `numeric_data` to extract the first two principal components and store them as `principal_components`.

*** =hint
- You can find the mean and standard deviation along each column by selecting `axis=0` in `np.mean` and `np.std`, respectively.
- The method that returns the top two principal components is `sklearn.decomposition.PCA(2)`.  Store this as `pca`.

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
pca = # Enter your code here!
principal_components = # Enter your code here!


```

*** =solution
```{python}
numeric_data = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, axis=0)

import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Plot the first two principal components.  Color the high and low quality wine as red and blue, respectively.  Are the two well separated by the first two principal components?

*** =hint
- `principal_components` is already ordered.  How can you index `principal_components` to draw the first two components?  Store these as `x` and `y`!

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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  We are now ready to fit the wine data to our kNN classifier.  Create a function `accuracy(predictions, outcomes)` that takes two lists of the same size, and returns percent of elements that are equal for the two lists.
-  Use `accuracy` to compare the percentage of similar elements in `x=np.array([1,2,3])` and `y=np.array([1,2,4])`.
-  Print your answer.

*** =hint
- The `==` operator will test for element-wise equality for numpy arrays (1 if equal, and 0 if not).  You can then use `np.mean` to find the fraction of these elements that are equal!
- Remember that `np.mean` will find the fraction of equal values, not the percentage.  For this, you must multiply by `100`!

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
    return np.mean(predictions == outcomes)*100

x=np.array([1,2,3])
y=np.array([1,2,4])

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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  The dataset remains stored as `data`.  Because most wines in the dataset are classified as low quality, a very simple classifier predicts that all wines are of low accuracy.  Use the accuracy function to calculate how many wines in the dataset are of low quality.

*** =hint
- The `accuracy` function should work just fine, with `0` as the first argument!
- Compare `0` with the `"high_quality"` column in `data`.  How can you do that?

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
    return np.mean(predictions == outcomes)*100
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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Use the scikit-learn classifier `KNeighborsClassifier`, to predict which wines are high and low quality, and store as `library_predictions`.  Is this predictor better than the simple classifier in Question 6?
- Use the `accuracy` function to compare `library_predictions` with the high quality wines.  Print your answer.

*** =hint
- A `KNeighborsClassifier` object will contain a `predict` method --- try that on `numeric_data`!
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
    return np.mean(predictions == outcomes)*100
```

*** =sample_code
```{python}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = # Enter your code here!


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
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")            
success_msg("Great work!  Yes, this is better!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:428c9e7854
## Exercise 8

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Unlike the `scikit-learn` function, our homemade kNN classifier does not take any shortcuts in calculating which neighbors are closest to each wine, so it is probably too slow to perform on a single computer.  Use the `random` library to select the seed `123`, and sample 10 row indexes from the dataset using `random.sample`.  Store this selection as `selection`.  Is our accuracy comparable to the library's function?

*** =hint
- Try sampling from the rows using `range(n_rows)`.

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
    return np.mean(predictions == outcomes)*100
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

In this homework, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality" and will use the k-Nearest Neighbors classifier to determine whether or not other information about the wine helps us correctly predict whether a new wine will be of high quality.

*** =instructions
-  Use our homemade kNN classifier `knn_predict` on this sampled dataset to predict wine quality for each value in `predictors[selection]` with `k=5`, and store as an `np.array` object called `my_predictions`.
-  Using the `accuracy` function, compare these results to the selected rows in the `high_quality` variable in `data`.  Store these results as `percentage`.
-  Print your answer.

*** =hint
- Use `knn_predict` for each value in `predictors[selection]`, with `predictors`, `outcomes` and `k=5` as additional parameters.  This can be done quickly using a `for` loop or a list comprehension!
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
    return np.mean(predictions == outcomes)*100
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)

```

*** =sample_code
```{python}
predictors = np.array(numeric_data)
outcomes = np.array(data["high_quality"])
my_predictions = # Enter your code here!
percentage = # Enter your code here!

```

*** =solution
```{python}
predictors = np.array(numeric_data)
outcomes = np.array(data["high_quality"])
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors[selection]])
percentage = accuracy(my_predictions, data.high_quality[selection])
print(percentage)


```

*** =sct
```{python}
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")  
test_object("percentage",
            undefined_msg = "Did you define `percentage`?",
            incorrect_msg = "It looks like `percentage` wasn't defined correctly.")
success_msg("Great work! Our accuracy is comparable to the library's function!")
```
