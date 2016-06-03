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

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
-  The personal ID for each individual is found in the column "pid" in the stored data. Read in and store the `key_vilno_1.csv` and `key_vilno_2.csv`, consisting of the personal IDs for villages 1 and 2, respectively.



*** =hint
-  Use `np.loadtxt` to load the two files.  You might want to store these as type `int` using the parameter `dtype=int`!

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

--- type:NormalExercise lang:python xp:100 skills:1 key:147facfc92
## Exercise 3

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Define a Python dictionaries with personal IDs as keys and one of that individual's covariates as values.  Complete this for the sex, caste, and religion covariates, for villages 1 and 2.

*** =hint
-  For a string `characteristic`, try `df1.set_index("pid")[characteristic]`, and use `.to_dict()` to convert this object to a `dict`.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
df = pd.read_stata("individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
```

*** =sample_code
```{python}
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()

sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
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

--- type:NormalExercise lang:python xp:100 skills:1 key:9f789af19a
## Exercise 4

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Let's consider how similar individuals in this network are.  Recall that homophily occurs when nodes that are neighbors in a network also share a characteristic, such as sex or caste.  Our measure will be the proportion of edges whose nodes in the pair share a characteristic.  How much homophily do we expect by chance?  If a characteristics were distributed completely randomly, the probability that two nodes share a characteristic is simply the product of chances that both, independently, have that characteristic.  To get the chances they share ANY characteristic, we then simply sum the chances of sharing a particular characteristic.  How can we do this for our dataset?  Create a function chance_homophily(characteristics) that takes a dict of characteristics (from Question 3) and computes the chance homophily for that characteristic.
- Use your function to compute the chance homophily for `hist = {1:3,2:1,3:1}`.

*** =hint
- Try using `counter` in the `collections` library.  The values in a `counter` objects can be divided by the total count to get the proportion of individuals in each unique category.
- Then, simply sum the squares of these probabilities!

*** =pre_exercise_code
```{python}
import numpy as np
```

*** =sample_code
```{python}
from collections import Counter
def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,\n
    specified as a dictionary, chars.
    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props = chars_counts / sum(chars_counts)
    return sum(chars_props**2)

hist = {1:3,2:1,3:1}
chance_homophily(hist)
```

*** =solution
```{python}
from collections import Counter
def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,\n
    specified as a dictionary, chars.
    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props = chars_counts / sum(chars_counts)
    return sum(chars_props**2)

hist = {1:3,2:1,3:1}
chance_homophily(hist)    
```

*** =sct
```{python}
test_function("chance_homophily",
              not_called_msg = "Make sure to call `chance_homophily`!",
              incorrect_msg = "Check your definition of `chance_homophily` again.")
test_object("",
            undefined_msg = "Did you define `hist`?",
            incorrect_msg = "It looks like `hist` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:6d28b82a47
## Exercise 5

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Use your function to compute the chance homophily for sex, caste, and religion In villages 1 and 2.  Is the chance homophily for any attribute very high for either village?

*** =hint
- Use `chance_homophily` on `sex1`, `caste1`, `religion1`; `sex2`, `caste2`, and `religion2`.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
df = pd.read_stata("individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
pid1 = np.loadtxt("key_vilno_1.csv", dtype=int)
pid2 = np.loadtxt("key_vilno_2.csv", dtype=int)
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()
sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
```

*** =sample_code
```{python}
print("Village 1 chance of same sex:", chance_homophily(sex1))
print("Village 1 chance of same caste:", chance_homophily(caste1))
print("Village 1 chance of same religion:", chance_homophily(religion1))

print("Village 2 chance of same sex:", chance_homophily(sex2))
print("Village 2 chance of same caste:", chance_homophily(caste2))
print("Village 2 chance of same religion:", chance_homophily(religion2))
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
test_function("chance_homophily",
              not_called_msg = "Make sure to call `chance_homophily`!",
              incorrect_msg = "Check your definition of `chance_homophily` again.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:49219b49be
## Exercise 6

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Now let's compute the actual homophily in our network.  Recall that our measure of homophily is the proportion of edges whose nodes in the pair share a characteristic.  Create a function `homophily(G, chars, IDs)` that takes a network, a dictionary of characteristics, and a list of personal IDs for the network, and outputs the homophily.

*** =hint
- For each edge in the network where both nodes have an ID, count up the fraction that have the same value of `chars`.  This is homophily!

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
def homophily(G, chars, IDs):
    """Given a network G, a dict of characteristics chars for node IDs,\n
    and dict of node IDs for each node in the network,\n
    find the homophily of the network."""
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

*** =solution
```{python}
def homophily(G, chars, IDs):
    """Given a network G, a dict of characteristics chars for node IDs,\n
    and dict of node IDs for each node in the network,\n
    find the homophily of the network."""
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
test_function("",
              not_called_msg = "Make sure to call ``!",
              incorrect_msg = "Check your definition of `` again.")
test_object("",
            undefined_msg = "Did you define ``?",
            incorrect_msg = "It looks like `` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:73e9f60471
## Exercise 7

Homophily is a network characteristic.  Homophily occurs when nodes that are neighbors in a network also share a characteristic more often than nodes that are not network neighbors.  In these exercises, we will investigate homophily of several characteristics of individuals connected in social networks in rural India.

*** =instructions
- Recall that Villages 1 and 2 have been stored as networkx networks `G1` and `G2`.  Use your function to compute the actual homophily for sex, caste, and religion in Villages 1 and 2.  Are these values higher or lower than that expected by chance?

*** =hint
- Use your `homophily` function on `sex1`, `caste1`, and `religion1` with `pid1` and `sex2`, `caste2`, and `religion2` with `pid2`.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
df = pd.read_stata("individual_characteristics.dta")
df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
pid1 = np.loadtxt("key_vilno_1.csv", dtype=int)
pid2 = np.loadtxt("key_vilno_2.csv", dtype=int)
sex1      = df1.set_index("pid")["resp_gend"].to_dict()
caste1    = df1.set_index("pid")["caste"].to_dict()
religion1 = df1.set_index("pid")["religion"].to_dict()
sex2      = df2.set_index("pid")["resp_gend"].to_dict()
caste2    = df2.set_index("pid")["caste"].to_dict()
religion2 = df2.set_index("pid")["religion"].to_dict()
```

*** =sample_code
```{python}
print("Village 1 actual proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 1 actual proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 1 actual proportion of same religion:", homophily(G1, religion1, pid1))

print("Village 2 actual proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 2 actual proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 2 actual proportion of same religion:", homophily(G2, religion2, pid2))
```

*** =solution
```{python}
print("Village 1 actual proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 1 actual proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 1 actual proportion of same religion:", homophily(G1, religion1, pid1))

print("Village 2 actual proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 2 actual proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 2 actual proportion of same religion:", homophily(G2, religion2, pid2))
```

*** =sct
```{python}
test_function("",
              not_called_msg = "Make sure to call `homophily`!",
              incorrect_msg = "Check your definition of `homophily` again.")
success_msg("Great work!")
```


