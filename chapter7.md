---
title       : Module 5 (Network) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
-  `individual_characteristics.dta` contains several characteristics for each individual in rural Indian social networks dataset, such as age, religion, and caste.  Use the pandas library to read in and store these characteristics as `df`.
-  We will focus only on villages 1 and 2:  Store separate datasets for individuals belonging to villages 1 and 2 as `df1` and `df2`, respectively.  (Note that some attributes may be missing for some individuals. Here, investigate only those pairs of nodes where the attributes are known for both nodes. This means that we're effectively assuming that the data are missing completely at random.)
- Use the `head` function to display the first few entries of `df1`.

*** =hint
- For reading in the dataset directly, try `pd.read_stata`!

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
import pandas as pd
df = pd.read_stata("individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
```

*** =solution
```{python}
import pandas as pd
df = pd.read_stata("individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
```

*** =sct
```{python}
#test_function("",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.")
test_object("df1",
            undefined_msg = "Did you define `df1`?",
            incorrect_msg = "It looks like `df1` wasn't defined correctly.")
test_object("df2",
            undefined_msg = "Did you define `df2`?",
            incorrect_msg = "It looks like `df2` wasn't defined correctly.")            
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:95681c66de
## Exercise 2

Description

*** =instructions
-  The personal ID for each individual is found in the column "pid" in the stored data. Read in and store the `key_vilno_1.csv` and `key_vilno_2.csv`, consisting of the personal IDs for villages 1 and 2, respectively.



*** =hint
-

*** =pre_exercise_code
```{python}
import numpy as np
```

*** =sample_code
```{python}
pid1 = np.loadtxt("key_vilno_1.csv", dtype=int)
pid2 = np.loadtxt("key_vilno_2.csv", dtype=int)
```

*** =solution
```{python}
pid1 = np.loadtxt("key_vilno_1.csv", dtype=int)
pid2 = np.loadtxt("key_vilno_2.csv", dtype=int)
```

*** =sct
```{python}
test_object("pid1",
            undefined_msg = "Did you define `pid1`?",
            incorrect_msg = "It looks like `pid1` wasn't defined correctly.")
test_object("pid2",
            undefined_msg = "Did you define `pid2`?",
            incorrect_msg = "It looks like `pid2` wasn't defined correctly.")
success_msg("Great work!")
```


