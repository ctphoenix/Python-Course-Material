---
title       : Case Study 6 - Social Network Analysis
description : Homophily is a network characteristic.  Homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
-  `individual_characteristics.dta` contains several characteristics for each individual in the dataset such as age, religion, and caste.  Use the `pandas` library to read in and store these characteristics as a dataframe called `df`.
-  Store separate datasets for individuals belonging to Villages 1 and 2 as `df1` and `df2`, respectively.
    -  Note that some attributes may be missing for some individuals. In this case study, we will ignore rows of data where some column information is missing.
- Use the `head` method to display the first few entries of `df1`.

*** =hint
-  `df["village"]==1` tests if each row belongs to Village 1.  How can you use this to subset the rows of `df` belonging to Village 1?
- Don't forget to call `df1.head()`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
import pandas as pd
df  = pd.read_stata(data_filepath + "individual_characteristics.dta")
df1 = # Enter code here!
df2 = # Enter code here!

# Enter code here!
```

*** =solution
```{python}
import pandas as pd
df  = pd.read_stata(data_filepath + "individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]

df1.head()
```

*** =sct
```{python}
test_object("df",
            undefined_msg = "Did you define `df`?",
            incorrect_msg = "It looks like `df` wasn't defined correctly.")
test_object("df1",
            undefined_msg = "Did you define `df1`?",
            incorrect_msg = "It looks like `df1` wasn't defined correctly.")
test_object("df2",
            undefined_msg = "Did you define `df2`?",
            incorrect_msg = "It looks like `df2` wasn't defined correctly.")   
test_student_typed("df1.head()",
              pattern=False,
              not_typed_msg="Did you call `df1.head()`?")            
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:95681c66de
## Exercise 2

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
-  In this dataset, each individual has a personal ID, or PID, stored in `key_vilno_1.csv` and `key_vilno_2.csv` for villages 1 and 2, respectively. `data_filepath` contains the base URL to the datasets used in this exercise. Use `pd.read_csv` to read in and store `key_vilno_1.csv` and `key_vilno_2.csv` as `pid1` and `pid2` respectively.  The `csv` files have no headers, so make sure to include the parameter `header = None`.

*** =hint
- Remember, you can concatenate strings with `+`. You can also take a look at the contents of string `data_filepath` using the shell.
-   You might want to store these as type `int` using the parameter `dtype=int`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
```

*** =sample_code
```{python}
# Enter code here!



```

*** =solution
```{python}
pid1 = pd.read_csv(data_filepath + "key_vilno_1.csv", dtype=int, header = None)
pid2 = pd.read_csv(data_filepath + "key_vilno_2.csv", dtype=int, header = None)
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

--- type:NormalExercise lang:python xp:100 skills:2 key:147facfc92
## Exercise 3

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Define dictionaries with personal IDs as keys and a given covariate for that individual as values.  Complete this for the sex, caste, and religion covariates, for Villages 1 and 2.
- For Village 1, store these dictionaries into variables named `sex1`, `caste1`, and `religion1`.
- For Village 2, store these dictionaries into variables named `sex2`, `caste2`, and `religion2`.

*** =hint



*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
df = pd.read_stata(data_filepath + "individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
```

*** =sample_code
```{python}
sex1      = # Enter code here!
caste1    = # Enter code here!
religion1 = # Enter code here!

# Continue for df2 as well.


```

*** =solution
```{python}
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()

sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
```

*** =sct
```{python}
test_object("sex1",
            undefined_msg = "Did you define `sex1`?",
            incorrect_msg = "It looks like `sex1` wasn't defined correctly.")
test_object("caste1",
            undefined_msg = "Did you define `caste1`?",
            incorrect_msg = "It looks like `caste1` wasn't defined correctly.")
test_object("religion1",
            undefined_msg = "Did you define `religion1`?",
            incorrect_msg = "It looks like `religion1` wasn't defined correctly.")
test_object("sex2",
            undefined_msg = "Did you define `sex2`?",
            incorrect_msg = "It looks like `sex2` wasn't defined correctly.")
test_object("caste2",
            undefined_msg = "Did you define `caste2`?",
            incorrect_msg = "It looks like `caste2` wasn't defined correctly.")
test_object("religion2",
            undefined_msg = "Did you define `religion2`?",
            incorrect_msg = "It looks like `religion2` wasn't defined correctly.")            
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:9f789af19a
## Exercise 4

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

Let's consider how much homophily exists in these networks.  For a given characteristic, our measure of homophily will be the proportion of edges in the network whose constituent nodes share that characteristic.  How much homophily do we expect by chance?  If characteristics are distributed completely randomly, the probability that two nodes `x` and `y` share characteristic `a` is the probability both nodes have characteristic `a`, which is the frequency of `a` squared.  The total probability that nodes `x` and `y` share their characteristic is therefore the sum of the frequency of each characteristic in the network.  For example, in the dictionary `favorite_colors` provided, the frequency of `red` and `blue` is 1/3 and 2/3 respectively, so the chance homophily is (1/3)^2+(2/3)^2 = 5/9.

*** =instructions
- Create a function `chance_homophily(chars)` that takes a dictionary with personal IDs as keys and characteristics as values, and computes the chance homophily for that characteristic.
- A sample of three peoples' favorite colors is given in `favorite_colors`.  Use your function to compute the chance homophily in this group, and store as `color_homophily`.
- Print `color_homophily`.

*** =hint
- Recall that the `Counter` method takes a `list` and creates a dictionary-like object with unique list values as keys and their counts as values.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import numpy as np
```

*** =sample_code
```{python}
from collections import Counter
def chance_homophily(chars):
    # Enter code here!

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

color_homophily = chance_homophily(favorite_colors)
print(color_homophily)

```

*** =solution
```{python}
from collections import Counter
def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,
    specified as a dictionary, chars.
    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props  = chars_counts / sum(chars_counts)
    return sum(chars_props**2)

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

color_homophily = chance_homophily(favorite_colors)
print(color_homophily)

```

*** =sct
```{python}
test_function("chance_homophily",
              not_called_msg = "Make sure to call `chance_homophily`!",
              incorrect_msg = "Check your definition of `chance_homophily` again.")
test_object("color_homophily",
            undefined_msg = "Did you define `color_homophily`?",
            incorrect_msg = "It looks like `color_homophily` wasn't defined correctly.")              
test_student_typed("print",
			  pattern=False,
              not_typed_msg = "Did you remember to print your answer?")             
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:6d28b82a47
## Exercise 5

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- `sex1`, `caste1`, `religion1`, `sex2`, `caste2`, and `religion2` are already defined from previous exercises.  Use `chance_homophily` to compute the chance homophily for sex, caste, and religion In Villages 1 and 2.  Is the chance homophily for any attribute very high for either village?

*** =hint
- Use `chance_homophily` on `sex1`, `caste1`, `religion1`, `sex2`, `caste2`, and `religion2`.
- Print all six values.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
df = pd.read_stata(data_filepath + "individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
pid1 = pd.read_csv(data_filepath + "key_vilno_1.csv", dtype=int, header = None)
pid2 = pd.read_csv(data_filepath + "key_vilno_2.csv", dtype=int, header = None)
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()
sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
from collections import Counter
def chance_homophily(chars):
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props  = chars_counts / sum(chars_counts)
    return sum(chars_props**2)
```

*** =sample_code
```{python}
print("Village 1 chance of same sex:", chance_homophily(sex1))
# Enter your code here.






```

*** =solution
```{python}
print("Village 1 chance of same sex:", chance_homophily(sex1))
print("Village 1 chance of same caste:", chance_homophily(caste1))
print("Village 1 chance of same religion:", chance_homophily(religion1))

print("Village 2 chance of same sex:", chance_homophily(sex2))
print("Village 2 chance of same caste:", chance_homophily(caste2))
print("Village 2 chance of same religion:", chance_homophily(religion2))
```

*** =sct
```{python}
test_student_typed("chance_homophily(sex1)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `sex1`?")              
test_student_typed("chance_homophily(caste1)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `caste1`?")  
test_student_typed("chance_homophily(religion1)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `religion1`?")  
test_student_typed("chance_homophily(sex2)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `sex2`?")  
test_student_typed("chance_homophily(caste2)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `caste2`?")  
test_student_typed("chance_homophily(religion2)",
              pattern=False,
              not_typed_msg = "Did you use `chance_homophily` for `religion2`?")
test_student_typed("print",
			  pattern=False,
              not_typed_msg = "Did you remember to print your answers?")              
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:49219b49be
## Exercise 6

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Now let's compute the observed homophily in our network.  Recall that our measure of homophily is the proportion of edges whose nodes share a characteristic.  `homophily(G, chars, IDs)` takes a network `G`, a dictionary of characteristics `chars`, and node IDs `IDs`.  For each node pair, determine whether a tie exists between them, as well as whether they share a characteristic.  The total count of these is `num_same_ties` and `num_ties` respectively, and their ratio is the homophily of `chars` in `G`.  Complete the function by choosing where to increment `num_same_ties` and `num_ties`.

*** =hint
- You can increment an `int` variable `x` using the Python shorthand `x += 1`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:   # do not double-count edges!
                if IDs[n1] in chars and IDs[n2] in chars:
                    if G.has_edge(n1, n2):
                        # Should `num_ties` be incremented?  What about `num_same_ties`?
                        if chars[IDs[n1]] == chars[IDs[n2]]:
                            # Should `num_ties` be incremented?  What about `num_same_ties`?
    return (num_same_ties / num_ties)
    
```

*** =solution
```{python}
def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:   # do not double-count edges!
                if IDs[n1] in chars and IDs[n2] in chars:
                    if G.has_edge(n1, n2):
                        num_ties += 1
                        if chars[IDs[n1]] == chars[IDs[n2]]:
                            num_same_ties += 1
    return (num_same_ties / num_ties)
```

*** =sct
```{python}
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:73e9f60471
## Exercise 7

Network homophily occurs when nodes that share an edge share a characteristic more often than nodes that do not share an edge.  In this case study, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- The networks for Villages 1 and 2 have been stored as `networkx` graph objects `G1` and `G2`.  Use your `homophily` function to compute the observed homophily for sex, caste, and religion in Villages 1 and 2.
- Print all six values.  Are these values higher or lower than that expected by chance?

*** =hint
- Use your `homophily` function on `sex1`, `caste1`, and `religion1` with `pid1`, and `sex2`, `caste2`, and `religion2` with `pid2`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import pandas as pd
import numpy as np
import networkx as nx
df = pd.read_stata(data_filepath + "individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
pid1 = np.array(pd.read_csv(data_filepath + "key_vilno_1.csv", dtype=int, header = None)[0])
pid2 = np.array(pd.read_csv(data_filepath + "key_vilno_2.csv", dtype=int, header = None)[0])
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()
sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
A1 = np.array(pd.read_csv(data_filepath + "adj_allVillageRelationships_vilno_1.csv", delimiter=",", header = None))
A2 = np.array(pd.read_csv(data_filepath + "adj_allVillageRelationships_vilno_2.csv", delimiter=",", header = None))
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)
def homophily(G, chars, IDs):
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:   # do not double-count edges!
                if IDs[n1] in chars and IDs[n2] in chars:
                    if G.has_edge(n1, n2):
                        num_ties += 1
                        if chars[IDs[n1]] == chars[IDs[n2]]:
                            num_same_ties += 1
    return (num_same_ties / num_ties)    
    
```

*** =sample_code
```{python}
print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
# Enter your code here!





```

*** =solution
```{python}
print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 1 observed proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 1 observed proportion of same religion:", homophily(G1, religion1, pid1))

print("Village 2 observed proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 2 observed proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 2 observed proportion of same religion:", homophily(G2, religion2, pid2))

```

*** =sct
```{python}
test_student_typed("sex1",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `sex1`?")              
test_student_typed("caste1",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `caste1`?")  
test_student_typed("religion1",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `religion1`?")  
test_student_typed("sex2",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `sex2`?")  
test_student_typed("caste2",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `caste2`?")  
test_student_typed("religion2",
              pattern=False,
              not_typed_msg="Did you use `homophily` for `religion2`?")               
success_msg("Great work!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```


