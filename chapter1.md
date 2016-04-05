---
title       : Homework 1
description : Exercises for homework (Week 1)
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1a

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- The lowercase English alphabet can be found using `ascii_lowercase` attribute in the `string` library.  Store this as `alphabet`.

*** =hint
- Use `import` to import the `string` library.
- Use `=` to assign `ascii_lowercase` to `alphabet`.

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!

import string
alphabet = string.ascii_lowercase

```

*** =solution
```{python}
import string
alphabet = string.ascii_lowercase
```

*** =sct
```{python}
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
#test_function("ascii_lowercase",
#              not_called_msg = "Make sure to use `ascii_lowercase`!",
#              incorrect_msg = "Check your usage of `ascii_lowercase` again.")
test_object("alphabet",
            undefined_msg = "Did you define `alphabet`?",
            incorrect_msg = "It looks like `alphabet` wasn't defined correctly.")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1b

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- Consider the sentence 'Jim quickly realized that the beautiful gowns are expensive'.  Create a dictionary `count_letters` with each letter used as keys and the number of times each letter is used in this sentence as values.  Make sure that capital letters are counted!

*** =hint
- Can you think of a way to use `for` loops to count up the frequency of each letter used in the sentence? 

*** =pre_exercise_code
```{python}
import string
alphabet = string.ascii_lowercase
```


*** =sample_code
```{python}
# write your code here!

sentence = 'Jim quickly realized that the beautiful gowns are expensive.'

count_letters = {}
for letter in alphabet:
    count_letter = 0
    for character in sentence:
        if character.casefold() == letter:
            count_letter += 1
    count_letters[letter] = count_letter

```

*** =solution
```{python}
sentence = 'Jim quickly realized that the beautiful gowns are expensive.'

count_letters = {}
for letter in alphabet:
    count_letter = 0
    for character in sentence:
        if character.casefold() == letter:
            count_letter += 1
    count_letters[letter] = count_letter
```

*** =sct
```{python}

test_object("sentence",
            undefined_msg = "Did you define `sentence`?",
            incorrect_msg = "It looks like `sentence` wasn't defined correctly.")
test_object("count_letters",
            undefined_msg = "Did you define `count_letters`?",
            incorrect_msg = "It looks like `count_letters` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1c

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- Here is an example solution of part b.)  Rewrite this code to make a function called `counter` that takes a string `input_string` and returns a dictionary of letter counts `count_letters`.  Use your function to call `counter(sentence)`.

*** =hint
- Add `def` at the beginning to define the function, indent the inner code, and use `return` at the end to ensure your function returns the output.

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
import string
sentence = 'Jim quickly realized that the beautiful gowns are expensive.'

# edit this code!
count_letters = {}
for letter in string.ascii_lowercase:
    count_letter = 0
    for character in input_string:
        if character.casefold() == letter:
            count_letter += 1
    count_letters[letter] = count_letter


```

*** =solution
```{python}
import string 
sentence = 'Jim quickly realized that the beautiful gowns are expensive.'

def counter(input_string):
    count_letters = {}
    for letter in string.ascii_lowercase:
        count_letter = 0
        for character in input_string:
            if character.casefold() == letter:
                count_letter += 1
        count_letters[letter] = count_letter
    return count_letters
    
counter(sentence)
```

*** =sct
```{python}
test_function("counter", index = 1,
              not_called_msg = "Make sure to use `ascii_lowercase`!",
              incorrect_msg = "Check your usage of `ascii_lowercase` again.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1d

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- In the course repository of the Abraham Lincoln's Gettysburg Address. Use your function from part b.) to count the number of letters in this address.

*** =hint
-  Read in the Gettysbug Address using `open`.  Can you use `counter` to do count the frequency of each letter?

*** =pre_exercise_code
```{python}
import string
def counter(input_string):
    count_letters = {}
    for letter in string.ascii_lowercase:
        count_letter = 0
        for character in input_string:
            if character.casefold() == letter:
                count_letter += 1
        count_letters[letter] = count_letter
    return count_letters
```

*** -sample_code
```{python}

```

*** =solution
```{python}
with open('gettysburg.txt', 'r') as f:
    address = f.read()
    address_count = counter(address)
    
print(address_count)
```

*** =sct
```{python}
test_object("address_count",
              not_called_msg = "Make sure to define `address_count`!",
              incorrect_msg = "Are you sure `address_count` is correct?")
test_function("print", index = 1,
              not_called_msg = "Make sure to use `print`!",
              incorrect_msg = "Check your usage of `print` again.")              
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1e

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- What is the most common letter used in the Gettysburg Address?  Print your answer.

*** =hint
- You will have to find the key that corresponds to the maximum value in `address_count`!

*** =pre_exercise_code
```{python}
import string
def counter(input_string):
    count_letters = {}
    for letter in string.ascii_lowercase:
        count_letter = 0
        for character in input_string:
            if character.casefold() == letter:
                count_letter += 1
        count_letters[letter] = count_letter
    return count_letters
    
with open('gettysburg.txt', 'r') as f:
address = f.read()
address_count = counter(address)
```

*** =solution
```{python}
import string
def counter(input_string):
    count_letters = {}
    for letter in string.ascii_lowercase:
        count_letter = 0
        for character in input_string:
            if character.casefold() == letter:
                count_letter += 1
        count_letters[letter] = count_letter
    return count_letters
    
with open('gettysburg.txt', 'r') as f:
address = f.read()
address_count = counter(address)

maximum, letter_maximum  = 0, ""
for letter in address_count.keys():
    if address_count[letter] > maximum:
        letter_maximum = letter

print(letter)
```

*** =sample_code
```{python}
# write your code here!

maximum, letter_maximum  = 0, ""
for letter in address_count.keys():
    if address_count[letter] > maximum:
        letter_maximum = letter

print(letter)

```

*** =sct
```{python}
test_object("letter",
              not_called_msg = "Make sure to define `letter`!",
              incorrect_msg = "Are you sure `letter` is defined correctly?")
test_function("print", index = 1,
              not_called_msg = "Make sure to use `print`!",
              incorrect_msg = "Check your usage of `print` again.")                
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2a

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- Using the math library, calculate and print the value of pi/4.

*** =hint
- The `math` library contains a float `pi` --- try using that!

*** =pre_exercise_code
```{python}
```

*** =solution
```{python}
import math
print(math.pi/4)
```

*** =sample_code
```{python}
# write your code here!
import math
print(math.pi/4)
```

*** =sct
```{python}
test_function("print", index = 1,
              not_called_msg = "Make sure to print your answer!",
              incorrect_msg = "What you printed is not yet correct.")
success_msg("Great work!")
```








--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2b

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- Using `random.random`, create a function `rand()` that generate real numbers between -1 and 1.
- Call `rand()` once.

*** =hint
-  `random.random` creates random numbers between 0 and 1.  How can you stretch and shift this range to get random values between 0 and 1?

*** =pre_exercise_code
```{python}
```

*** =solution
```{python}
import random
random.seed(1)
def rand():
    """
        Generates a random real number between -1 and 1.\n
        This function uses random.random, which generates\n
        random real number between 0 and 1.
    """
    return random.random()*2-1

rand()
```

*** =sample_code
```{python}
# write your code here!
import random
random.seed(1)
def rand():
    """
        Generates a random real number between -1 and 1.\n
        This function uses random.random, which generates\n
        random real number between 0 and 1.
    """
    return random.random()*2-1

rand()
```

*** =sct
```{python}
test_function("rand", index = 1,
              not_called_msg = "Make sure to call `rand()`!",
              incorrect_msg = "Did you define `rand` correctly?")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2c

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- The formula for the distance between two points `x` and `y` is `sqrt(sum((x-y)**2))`.  Create a function `distance(x, y)` that takes two vectors and outputs the distance between them.  Use your function to find the distance between `(0,0)` and `1,1`.
*** =hint
-

*** =pre_exercise_code
```{python}
import random
random.seed(1)
```

*** =solution
```{python}
import math
def distance(x, y):
    """
        Given x and y, find their distance.\n
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))

distance((0,0),(1,1))
```

*** =sample_code
```{python}
# write your code here!
import math
def distance(x, y):
    """
        Given x and y, find their distance.\n
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))

distance((0,0),(1,1))
```

*** =sct
```{python}
test_function("distance", index = 1,
              not_called_msg = "Did you use your `distance` function?",
              incorrect_msg = "Are you sure that is the correct distance?")
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2d

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- Make a function `in_circle(x)` that determines if a two-dimensional point falls within the the unit circle.  That is, find if a two-dimensional point has distance `<1` from the origin `(0,0)`.  Use your function to find whether the point `(1,1)` lies within the unit circle centered at the origin.

*** =hint
- Use your previous function `distance` to test if the distance between the point and `(0,0)` is less than 1!

*** =pre_exercise_code
```{python}
import random, math
random.seed(1)
def distance(x, y):
    """
        Given x and y, find their distance.\n
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
def distance(x, y):
    """
        Given x and y, find their distance.\n
        This is given by sqrt(sum((x-y)**2)).
    """
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i]-y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))
def in_circle(x, origin = [0]*2):
    """
        This function determines if a two-dimensional point\n
        falls within the\n unit circle.
    """
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False

in_circle((1,1))
```

*** =sample_code
```{python}
# write your code here!
def in_circle(x, origin = [0]*2):
    """
        This function determines if a two-dimensional point\n
        falls within the\n unit circle.
    """
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False

in_circle((1,1))
```

*** =sct
```{python}
test_function("in_circle", index = 1,
              not_called_msg = "Did you use your `in_circle` function?",
              incorrect_msg = "Is the output of `in_circle` correct?")
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2e

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- Find how many of `R=1000` two-dimensional points selected at random from `[-1,1]^2` fall within the unit circle, and print your answer.  This proportion is an estimate of the ratio of the two volumes!

*** =hint
-  Use your functions `rand()` and `in_circle()` to create 1000 points, test if they fall within the unit circle.  Make sure to print the fraction that do fit inside the circle!

*** =pre_exercise_code
```{python}
import random, math
random.seed(1)
def rand():
    return random.random()*2-1
def distance(x, y):
    """
        Given x and y, find their distance.\n
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
import random, math
random.seed(1)
def rand():
    return random.random()*2-1
def distance(x, y):
    """
        Given x and y, find their distance.\n
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
R=1000
inside = 0
for i in range(R):
    x = [rand(), rand()]
    inside += int(in_circle(x))
print(inside/R)
```

*** =sample_code
```{python}
# write your code here!
R=1000
inside = 0
for i in range(R):
    x = [rand(), rand()]
    inside += int(in_circle(x))
print(inside/R)
```

*** =sct
```{python}
test_function("print", index = 1,
              not_called_msg = "Make sure to print your answer!",
              incorrect_msg = "Are you sure that your answer is correct?")
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 2f

The ratio of the volumes of a circle and the square inscribing it is `pi/4`.  In this exercise, we will find a way to approximate this value.

*** =instructions
- Recall the true ratio of the volume of of the unit circle to the volume to the inscribing square is pi/4. Find and print the difference between this value and your estimate in part `2e`.

*** =hint
- Take your estimate from the last exercise, and subtract `math.pi/4`.  Make sure to print your answer!

*** =pre_exercise_code
```{python}
import random, math
random.seed(1)
def rand():
    return random.random()*2-1
def distance(x, y):
    """
        Given x and y, find their distance.\n
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
R=10000
inside = 0
for i in range(R):
    x = [rand(), rand()]
    inside += int(in_circle(x))        
```

*** =solution
```{python}
import random, math
random.seed(1)
def rand():
    return random.random()*2-1
def distance(x, y):
    """
        Given x and y, find their distance.\n
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
R=10000
inside = 0
for i in range(R):
    x = [rand(), rand()]
    inside += int(in_circle(x))         
print(inside/R - math.pi/4)
```

*** =sample_code
```{python}
# write your code here!
print(inside/R - math.pi/4)
```

*** =sct
```{python}
test_function("print", index = 1,
              not_called_msg = "Make sure to print your answer!")
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 3a

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  One way to smooth it out is to consider the average of each value's neighbors, including the value itself.  

*** =instructions
- Make a function `moving_window_average(x, n_neighbors)` that takes a list `x` and the number of neighbors `n_neighbors` on either side to consider. For each value, the function computes the average of each value's neighbors, including themselves. Have the function return a list of these averaged values as long as the original list.  If there are not enough neighbors (for cases near the edge), substitute the original value as many times as there are missing neighbors.
- Use your function to find the moving window sum of `x=[0,10,5,3,1,5]` and `n_neighbors=1`.

*** =hint
- First concatenate the two edges of your list with repeats of the first and last values of the list, and use a moving sum that applies to the middle values of this longer list!

*** =pre_exercise_code
```{python}
import random
random.seed(1)
```

*** =solution
```{python}
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]
moving_window_average(x, 1)
```

*** =sample_code
```{python}
# write your code here!
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]
moving_window_average(x, 1)    
```

*** =sct
```{python}
test_object("x",
            undefined_msg = "Did you remember to define `x`?",
            incorrect_msg = "It looks like `x` wasn't defined correctly.")
test_function("moving_window_average", index = 1,
              not_called_msg = "Make sure to use `moving_window_average`!",
              incorrect_msg = "Are you sure that your answer is correct?")
success_msg("Great work!")
```







--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 3b

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  One way to smooth it out is to consider the average of each value's neighbors, including the value itself.

*** =instructions
- Compute and store `R=1000` random values from 0-1 as `x`. Then, compute the moving window average several times for this list for the range of number of neighbors 1-9.  Store x and each of these averages as consecutive lists in a list called `X`.

*** =hint
- You may be able to use a list comprehension here!  A `for` loop will also work.

*** =pre_exercise_code
```{python}
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]
```

*** =solution
```{python}
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

R=1000
x = [random.random() for i in range(R)]
X = [x] + [moving_window_sum(x, i) for i in range(1,10)]
```

*** =sample_code
```{python}
# write your code here!
R=1000
x = [random.random() for i in range(R)]
X = [x] + [moving_window_sum(x, i) for i in range(1,10)]
```

*** =sct
```{python}
test_object("X",
              undefined_msg="Make sure to define `X`!",
              incorrect_msg="Check your usage of `X` again.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 3c

A list of numbers can be very unsmooth, meaning very high numbers can be right next to very low numbers.  One way to smooth it out is to consider the average of each value's neighbors, including the value itself.

*** =instructions
- For each list in `X`, calculate and store the range (the maximum minus the minimum) in a new list `ranges`, and print.  As the moving average window increases, does the range of each list increase or decrease? Why do you think that is?

*** =hint
- Another `for` loop

*** =pre_exercise_code
```{python}
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]

moving_window_sum(x, n_neighbors=1)

R=1000
x = [random.random() for i in range(R)]
X = [x] + [moving_window_sum(x, i) for i in range(1,10)]
```

*** =solution
```{python}
import random
random.seed(1)
def moving_window_average(x, n_neighbors=2):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[n-1]]*n_neighbors
    return [sum(x[i:(i+width)]) / width for i in range(n)]

x=[0,10,5,3,1,5]

moving_window_saverage(x, n_neighbors=1)

R=1000
x = [random.random() for i in range(R)]
X = [x] + [moving_window_average(x, i) for i in range(1,10)]
ranges = [max(x)-min(x) for x in X]
print(ranges)
```

*** =sample_code
```{python}
# write your code here!
ranges = [max(x)-min(x) for x in X]
print(ranges)
```

*** =sct
```{python}
test_object("ranges",
            undefined_msg = "Did you remember to define `ranges`?",
            incorrect_msg = "It looks like `ranges` wasn't defined correctly.")
test_function("print", index = 1,
              not_called_msg = "Make sure to print `ranges`!",
              incorrect_msg = "Are you sure that your answer is correct?")
success_msg("Great work!  The range decreases, because the average smooths a larger number of neighbors. Because the numbers in the original list are just random, we expect the average of many of them to be roughly 1/2, and more averaging means more smoothness in this value.")
```




