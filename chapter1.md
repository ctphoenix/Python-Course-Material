---
title       : Homework 1
description : Exercises for homework (Week 1).  In this homework, we will use objects, functions, and randomness to find the length of documents, approximate pi, and smooth out random noise.
--- type:NormalExercise lang:python xp:100 skills:2 key:e637b6eee9
## Exercise 1a

In this five-part exercise, we will count the frequency of each letter in a given string.

*** =instructions
- Import the `string` library.
- Create a variable `alphabet` that consists of the lowercase and uppercase letters in the English alphabet using the `ascii_letters` attribute of the `string` library.

*** =hint
- Use `import` to import the `string` library.
- Use `=` to assign `ascii_letters` to `alphabet`.


*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
# Write your code here!



```

*** =solution
```{python}
import string
alphabet = string.ascii_letters
```

*** =sct
```{python}
test_student_typed("ascii_letters",
                       pattern=False,
                       not_typed_msg="Make sure to use `ascii_letters`!")
test_object("alphabet",
            undefined_msg = "Did you define `alphabet`?",
            incorrect_msg = "It looks like `alphabet` wasn't defined correctly.")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:89cba9d6a8
## Exercise 1b

In this five-part exercise, we will count the frequency of each letter in a given string.

*** =instructions
- The lower and upper cases of the English alphabet is stored as `alphabet`.
- Consider the sentence 'Jim quickly realized that the beautiful gowns are expensive'.  Create a dictionary `count_letters` with keys consisting of each unique letter in the sentence and values consisting of the number of times each letter is used in this sentence.  Count both upper case and lower case letters separately in the dictionary.

*** =hint
- Can you think of a way to use a `for` loop to count up the frequency of each letter used in the sentence? 

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_letters
```


*** =sample_code
```{python}
sentence = 'Jim quickly realized that the beautiful gowns are expensive'

count_letters = {}
#write your code here!



```

*** =solution
```{python}
sentence = 'Jim quickly realized that the beautiful gowns are expensive'

count_letters = {}
for letter in sentence:
    if letter in alphabet:
        if letter in count_letters:
            count_letters[letter] += 1
        else:
            count_letters[letter] = 1
```

*** =sct
```{python}
test_object("count_letters",
            undefined_msg = "Did you define `count_letters`?",
            incorrect_msg = "It looks like `count_letters` does count the letters in `sentence` correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:8cb1c4bf90
## Exercise 1c

In this five-part exercise, we will count the frequency of each letter in a given string.

*** =instructions
- Rewrite your code from  **1b** to make a function called `counter` that takes a string `input_string` and returns a dictionary of letter counts `count_letters`.  If you were unable to complete **1b**, you can use the solution by selecting `Show Answer`.
- Use your function to call `counter(sentence)`.

*** =hint
- Add `def` at the beginning to define the function, indent the inner code, and use `return` at the end to ensure your function returns the output.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_letters
```

*** =sample_code
```{python}
sentence = 'Jim quickly realized that the beautiful gowns are expensive'

# Create your function here!




```

*** =solution
```{python}
import string 

sentence = 'Jim quickly realized that the beautiful gowns are expensive'

def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in alphabet:
            if letter in count_letters:
                count_letters[letter] += 1
            else:
                count_letters[letter] = 1
    return count_letters
    
counter(sentence)
```

*** =sct
```{python}
test_function("counter", index = 1,
              not_called_msg = "Make sure to call `counter`!",
              incorrect_msg = "It appears that `counter` was not defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:a0932fb3c4
## Exercise 1d

In this five-part exercise, we will count the frequency of each letter in a given string.

*** =instructions
- Abraham Lincoln was a president during the American Civil War.  His famous 1863 Gettysburg Address has been stored as `address`, and the `counter` function defined in part **1c** has been loaded.  Use these to return a dictionary consisting of the count of each letter in this address, and save this as `address_count`.
- Print `address_count`.

*** =hint
-  Can you use `counter` to return the frequency of each letter?

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_letters

def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in alphabet:
            if letter in count_letters:
                count_letters[letter] += 1
            else:
                count_letters[letter] = 1
    return count_letters
import pandas as pd    
address = str(pd.read_csv(data_filepath + 'gettysburg.txt', error_bad_lines = False))    
```

*** =sample_code
```{python}
# Write your code here!



```

*** =solution
```{python}
address_count = counter(address)
print(address_count)

```

*** =sct
```{python}
test_object("address_count",
              undefined_msg = "Make sure to define `address_count`!",
              incorrect_msg = "Are you sure `address_count` is correct?")
test_function("print", index = 1,
              not_called_msg = "Make sure to use `print`!",
              incorrect_msg = "Check your usage of `print` again.")              
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:bb70667667
## Exercise 1e

In this five-part exercise, we will count the frequency of each letter in a given string.

*** =instructions
- The frequency of each letter in the Gettysburg Address is already stored as `address_count`.  Use this dictionary to find the most common letter in the Gettysburg address.
- Store this letter as `most_frequent_letter`, and print your answer.

*** =hint
- You will have to find the key that corresponds to the maximum value in `address_count`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"

import string
alphabet = string.ascii_letters

def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in alphabet:
            if letter in count_letters:
                count_letters[letter] += 1
            else:
                count_letters[letter] = 1
    return count_letters
    
import pandas as pd    
address = str(pd.read_csv(data_filepath + 'gettysburg.txt', error_bad_lines = False))    
address_count = counter(address)
```

*** =solution
```{python}
maximum = 0
letter_maximum = ""
for letter in address_count:
    if address_count[letter] > maximum:
        maximum = address_count[letter]
        most_frequent_letter = letter

print(most_frequent_letter)
```

*** =sample_code
```{python}
# write your code here!



```

*** =sct
```{python}
test_object("most_frequent_letter",
              undefined_msg = "Make sure to define `most_frequent_letter`!",
              incorrect_msg = "Are you sure `most_frequent_letter` is defined correctly?")
test_function("print", index = 1,
              not_called_msg = "Make sure to use `print`!",
              incorrect_msg = "Check your usage of `print` again.")                
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:8b40a3f09f
## Exercise 2a

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- Using the math library, calculate and print the value of pi / 4.

*** =hint
- The `math` library contains a float `pi` --- try using that!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =solution
```{python}
import math
print(math.pi / 4)
```

*** =sample_code
```{python}
# write your code here!



```

*** =sct
```{python}
test_student_typed("math.pi",
              pattern=True,
              not_typed_msg="Did you use `pi` from the `math` library?")
test_function("print", index = 1,
              not_called_msg = "Make sure to print your answer!",
              incorrect_msg = "What you printed is not yet correct.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:7aa7dadeac
## Exercise 2b

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- Using `random.uniform`, create a function `rand()` that generates a single `float` between `-1` and `1`.
- Call `rand()` once.  So we can check your solution, we will use `random.seed` to fix the value called by your function.

*** =hint
-  `random.uniform` generates a random value between the first argument and the second argument.  Try using this to get random values between `-1` and `1`.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =solution
```{python}
import random

random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.

def rand():
    """
        Generates a random real number between -1 and 1.
        This function uses random.uniform, which generates
        random real number between its first and second
        arguments.
    """
    return random.uniform(-1,1)

rand()
```

*** =sample_code
```{python}
import random

random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.

def rand():
   # define `rand` here!

rand()
```

*** =sct
```{python}
test_function("rand", index = 1,
              not_called_msg = "Make sure to call `rand()`!")
test_student_typed("random.uniform",
              pattern=False,
              not_typed_msg="Did you use `random.uniform` to generate your answer?")                            
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:eabc2b80ff
## Exercise 2c

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- The distance between two points `x` and `y` is the square root of the sum of squared differences along each dimension of `x` and `y`.  Create a function `distance(x, y)` that takes two vectors and outputs the distance between them.  Use your function to find the distance between `x=(0,0)` and `y=(1,1)`.
- Print your answer.

*** =hint
- Use the `sqrt` function in the `math` library to find square roots.  Finding the square can be done using the `pow` function in the `math` library, or exponentiating using `**2` after the number you would like to square.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
random.seed(1)
```

*** =solution
```{python}
import math

def distance(x, y):
    """
        Given x and y, find their distance.
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))

print(distance((0,0),(1,1)))
```

*** =sample_code
```{python}
import math

def distance(x, y):
   # define your function here!
   
   
   
```

*** =sct
```{python}
test_function("distance", index = 1,
              not_called_msg = "Did you use your `distance` function?",
              incorrect_msg = "Are you sure that is the correct distance?")
test_function("print", index = 1,
              not_called_msg = "Did you print your output?",
              incorrect_msg = "It appears what you have printed is incorrect.")              
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:2 key:89310e2304
## Exercise 2d

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- `distance(x, y)` is pre-loaded from part `2c`. Write a function `in_circle(x, origin)` that determines whether a two-dimensional point falls within a unit circle surrounding a given origin. Your function should return a boolean that is `True` if the distance between `x` and `origin` is less than 1, and `False` otherwise.
- Use your function to determine whether the point (1,1) lies within the unit circle centered at `(0,0)`.
- Print your answer.

*** =hint
- Use your previous function `distance` to test if the distance between the point and `(0,0)` is less than 1!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random, math

random.seed(1)

def distance(x, y):
    """
        Given x and y, find their distance.
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))
```

*** =solution
```{python}
import random, math

random.seed(1)
        
def in_circle(x, origin = [0]*2):
    """
        This function determines if a two-dimensional point
        falls within the unit circle.
    """
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False

print(in_circle((1,1)))
```

*** =sample_code
```{python}
import random, math

random.seed(1)

def in_circle(x, origin = [0]*2):
   # Define your function here!
   
   

```

*** =sct
```{python}
test_function("in_circle", index = 1,
              not_called_msg = "Did you use your `in_circle` function?",
              incorrect_msg = "Is the output of `in_circle` correct?")  
test_function("print", index = 1,
              not_called_msg = "Did you make sure to print your answer?",
              incorrect_msg = "It appears what you've printed is not correct.")  
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:2 key:c7c20038ea
## Exercise 2e

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- Create a list of `R=10000` booleans called `inside` that determines whether each point in `x` falls within the unit circle centered at `(0,0)`.  Make sure to use `in_circle`.
- Find the proportion of points within the circle by summing the count of `True` in `inside`, and dividing by `10000`.
- Print your answer.  This proportion is an estimate of the ratio of the two areas!

*** =hint
-  Use your functions `rand()` and `in_circle((x,y))` are pre-loaded from previous parts.  
- Print the fraction that do fit inside the circle!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random, math

random.seed(1)

def rand():
    return random.uniform(-1,1)
    
def distance(x, y):
    """
        Given x and y, find their distance.
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))    
        
def in_circle(x, origin = [0]*2):
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False
        
```

*** =solution
```{python}
R = 10000
x = []
inside = []
for i in range(R):
    point = [rand(), rand()]
    x.append(point)
    inside.append(in_circle(point))

print(sum(inside) / R)
```

*** =sample_code
```{python}
R = 10000
x = []
inside = []
for i in range(R):
    point = [rand(), rand()]
    x.append(point)
    # Enter your code here! #

# Enter your code here! #

```

*** =sct
```{python}
test_student_typed("print",
              pattern=False,
              not_typed_msg = "Make sure to print your answer!")
test_student_typed("in_circle",
                       pattern=False,
                       not_typed_msg="Make sure to use `in_circle()`!")                  
test_object("inside",
            undefined_msg = "Did you define `inside`?",
            incorrect_msg = "It looks like `inside` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:d3950c2ef5
## Exercise 2f

The ratio of the areas of a circle and the square inscribing it is `pi / 4`.  In this six-part exercise, we will find a way to approximate this value.

*** =instructions
- Note: `inside` and `R` are defined as in Exercise `2e`.  Recall that the true ratio of the area of the unit circle to the area to the inscribing square is `pi / 4`.
- Find the difference between your estimate from part `2e` and `math.pi / 4`.
- Print your answer.



*** =hint
- Take your estimate from the last exercise, and subtract `math.pi / 4`.  Make sure to print your answer!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random, math

random.seed(1)

def rand():
    return random.uniform(-1,1)
    
def distance(x, y):
    """
        Given x and y, find their distance.
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences)) 
        
def in_circle(x, origin = [0]*2):
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False
        
R = 10000
x = []
inside = []
for i in range(R):
    point = [rand(), rand()]
    x.append(point)
    inside.append(in_circle(point))       
```

*** =solution
```{python}
print(math.pi / 4 - sum(inside) / R)
```

*** =sample_code
```{python}
# write your code here!



```

*** =sct
```{python}
test_function("print", index = 1,
              incorrect_msg = "It appears what you've printed is not correct.", 
              not_called_msg = "Make sure to print your answer!")              
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:807ffbdc6f
## Exercise 3a

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  This list may represent a smooth path in reality that is masked with random noise (for example, satellite trajectories with inaccurate transmission). One way to smooth the values in the list is to replace each value with the average of each value's neighbors, including the value itself.  

*** =instructions
- Write a function `moving_window_average(x, n_neighbors)` that takes a list `x` and the number of neighbors `n_neighbors` on either side of a given member of the list to consider.
- For each value in `x`, `moving_window_average(x, n_neighbors)` computes the average of that value's neighbors, where neighbors includes the value itself.
- `moving_window_average` should return a list of averaged values that is the same length as the original list.
- If there are not enough neighbors (for cases near the edge), substitute the original value as many times as there are missing neighbors.
- Use your function to find the moving window sum of `x=[0,10,5,3,1,5]` and `n_neighbors=1`.


*** =hint
- First concatenate the two edges of your list with repeats of the first and last values of the list, and use a moving sum that applies to the middle values of this longer list!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
random.seed(1)
```

*** =solution
```{python}
import random

random.seed(1)

def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]
print(moving_window_average(x, 1))
```

*** =sample_code
```{python}
import random

random.seed(1)

def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    # To complete the function,
    # return a list of the mean of values from i to i+width for all values i from 0 to n-1.

x=[0,10,5,3,1,5]
print(moving_window_average(x, 1))
```

*** =sct
```{python}
test_object("x",
            undefined_msg = "Did you remember to define `x`?",
            incorrect_msg = "It looks like `x` wasn't defined correctly.")
test_function("moving_window_average", index = 1,
              not_called_msg = "Make sure to use `moving_window_average`!",
              incorrect_msg = "Are you sure that your answer is correct?")
test_function("print", index = 1,
              not_called_msg = "Did you remember to print your output?",
              incorrect_msg = "Are you sure that your answer is correct?")              
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:006c8d659a
## Exercise 3b

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  This list may represent a smooth path in reality that is masked with random noise (for example, satellite trajectories with inaccurate transmission). One way to smooth the values in the list is to replace each value with the average of each value's neighbors, including the value itself.  

*** =instructions
- Compute and store `R=1000` random values from 0-1 as `x`.
- `moving_window_average(x, n_neighbors)` is pre-loaded into memory from `3a`.  Compute the moving window average for `x` for values of `n_neighbors` ranging from 1 to 9 inclusive.
- Store `x` as well as each of these averages as consecutive lists in a list called `Y`. 


*** =hint
- You may be able to use a list comprehension here!  A `for` loop will also work.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
random.seed(1)
def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]
```

*** =solution
```{python}
import random

random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.
    
R=1000
x = [random.random() for i in range(R)]
Y = [x] + [moving_window_average(x, i) for i in range(1, 10)]
```

*** =sample_code
```{python}
import random

random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.
    
# write your code here!




```

*** =sct
```{python}
test_object("Y",
              undefined_msg="Make sure to define `Y`!",
              incorrect_msg="Check your usage of `Y` again.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:7f5873e828
## Exercise 3c

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  This list may represent a smooth path in reality that is masked with random noise (for example, satellite trajectories with inaccurate transmission). One way to smooth the values in the list is to replace each value with the average of each value's neighbors, including the value itself.  

*** =instructions
- `moving_window_average(x, n_neighbors=2)` and `Y` are already loaded into memory.  For each list in `Y`, calculate and store the range (the maximum minus the minimum) in a new list `ranges`.
- Print your answer.  As the window width increases, does the range of each list increase or decrease? Why do you think that is?

*** =hint
- A `for` loop or a list comprehension will work well here.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]

moving_window_average(x, n_neighbors=2)

R=1000
x = [random.random() for i in range(R)]
Y = [x] + [moving_window_average(x, i) for i in range(1,10)]
```

*** =solution
```{python}
ranges = [max(x)-min(x) for x in Y]
print(ranges)
```

*** =sample_code
```{python}
# write your code here!



```

*** =sct
```{python}
test_object("ranges",
            undefined_msg = "Did you remember to define `ranges`?",
            incorrect_msg = "It looks like `ranges` wasn't defined correctly.")
test_function("print", index = 1,
              not_called_msg = "Make sure to print `ranges`!",
              incorrect_msg = "Are you sure that your answer is correct?")
success_msg("Great work!  The range decreases, because the average smooths a larger number of neighbors. Because the numbers in the original list are just random, we expect the average of many of them to be roughly 1 / 2, and more averaging means more smoothness in this value.")
```




