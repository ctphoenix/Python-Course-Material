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
- The dataset contains a variable called `is_red`, making `color` redundant. Drop the color variable from the dataset, and save the new dataset as `numeric_data`.

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
#test_function("print",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("numeric_data",
            undefined_msg = "Did you define `numeric_data`?",
            incorrect_msg = "It looks like `numeric_data` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1 key:8515d59a47
## Exercise 3

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
-  To ensure that each variable contributes equally to the KNN classifier, standardize the data.  That is, subtract each variable by its mean, and divide it by its standard deviation.  Store this again as `numeric_data`.
- Principal components is a way to take a linear snapshot of the data from several different angles, with each snapshot ordered by how well they separate the data. The following code uses the scikit-learn (sklearn) library to find and store the two most informative angles, or principal components, of the data (a matrix with two columns corresponding to the principal components). Use this on your dataset to find and store the two principal components as `principal_components`.

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
numeric_data = data.drop("color", axis=1)
```

*** =sample_code
```{python}
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)

import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)
```

*** =solution
```{python}
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)

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

--- type:NormalExercise lang:python xp:100 skills:1 key:1b705ec875
## Exercise 4

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
-  Plot the first two principal components.  Color the high and low quality wine as red and blue, respectively.  Are the two well separated by the first two principal components?

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

plt.title("Principal Components of Wine") #1
plt.scatter(principal_components[:,0], principal_components[:,1], alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none') #2
plt.xlim(-8, 8); plt.ylim(-8, 8) #3
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2") #4
plt.show()
```

*** =solution
```{python}
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])

plt.title("Principal Components of Wine") #1
plt.scatter(principal_components[:,0], principal_components[:,1], alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none') #2
plt.xlim(-8, 8); plt.ylim(-8, 8) #3
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2") #4
plt.show()
```

*** =sct
```{python}
test_function("plt.show", # Use pythonwhat to do this!
              not_called_msg = "Make sure to call `plt.show`!",
              incorrect_msg = "Check your definition of `plt.show` again.")
#test_object("",
#            undefined_msg = "Did you define ``?",
#            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!  Yes, these differ significantly.")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:5817bdff2e
## Exercise 5

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
-  We are now ready to fit the wine data to our KNN classifier.  Create a function accuracy(predictions, outcomes) that takes two lists of the same size, and outputs the fraction of elements that are equal for the two lists.
-  Use `accuracy` to compare the percentage of similar elements in `x=np.array([1,2,3])` and `y=np.array([1,2,4])`.

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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)        
```

*** =sample_code
```{python}
def accuracy(predictions, outcomes):
    print(np.mean(predictions == outcomes)*100)

x=np.array([1,2,3])
y=np.array([1,2,4]))

accuracy(x,y)
```

*** =solution
```{python}
def accuracy(predictions, outcomes):
    """
    Finds the percent of predictions that equal outcomes.
    """
    print(np.mean(predictions == outcomes)*100)

x=np.array([1,2,3])
y=np.array([1,2,4])

accuracy(x,y)    
```

*** =sct
```{python}
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
test_object("",
            undefined_msg = "Did you define ``?",
            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:533b201c9b
## Exercise 6

In these exercises, we will analyse a dataset consisting of many different wines classified into "high quality" and "low quality", and will use K-nearest neighbors to predict whether or not other information about the wine helps us correctly guess whether a new wine will be of high quality.

*** =instructions
-  Because most wines are classified as low quality, a very simple classifier predicts that all wines are of low accuracy.  Use the accuracy function to calculate how many wines in the dataset are of low quality.

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
numeric_data = data.drop("color", axis=1)
numeric_data = (numeric_data - np.mean(numeric_data, 0)) / np.std(numeric_data, 0)
import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)    
def accuracy(predictions, outcomes):
    print(np.mean(predictions == outcomes)*100)    
```

*** =sample_code
```{python}
accuracy(0, data["high_quality"])
```

*** =solution
```{python}
accuracy(0, data["high_quality"])
```

*** =sct
```{python}
test_function("accuracy",
              not_called_msg = "Make sure to call `accuracy`!",
              incorrect_msg = "Check your definition of `accuracy` again.")
#test_object("",
#            undefined_msg = "Did you define ``?",
#            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!")
```
