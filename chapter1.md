---
title       : Homework 1
description : Exercises for homework (Week 1)
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1

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
## Exercise 1

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- Consider the sentence 'Jim quickly realized that the beautiful gowns are expensive'.  Create a dictionary that counts the number of times each letter is used in this sentence.  Make sure that capital letters are counted!

*** =hint
-

*** =pre_exercise_code
```{python}
```


*** =sample_code
```{python}
# write your code here!



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

test_object("count_letters",
            undefined_msg = "Did you define `count_letters`?",
            incorrect_msg = "It looks like `count_letters` wasn't defined correctly.")
test_object("count_letters",
            undefined_msg = "Did you define `sentence`?",
            incorrect_msg = "It looks like `sentence` wasn't defined correctly.")
success_msg("Great work!")
```





--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- Rewrite your code in part a.) as a function.  That is, make a function that takes a string and returns a dictionary of letter counts.

*** =hint
-

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!



```

*** =solution
```{python}
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

*** =sct
```{python}
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
test_function("counter",
              not_called_msg = "Make sure to define `counter`!",
              incorrect_msg = "Check your usage of `counter` again.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- In the course repository of the Abraham Lincoln's Gettysburg Address. Use your function from part b.) to count the number of letters in this address.

*** =hint
-

*** =pre_exercise_code
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
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
test_object("address_count",
              not_called_msg = "Make sure to define `address_count`!",
              incorrect_msg = "Check your usage of `address_count` again.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1

In this exercise, we will count the frequency of each letter in a document.

*** =instructions
- What is the most common letter used in the Gettysburg Address?  Print your answer.

*** =hint
-

*** =pre_exercise_code
```{python}
```

*** =solution
```{python}
maximum, letter_maximum  = 0, ""
for letter in address_count.keys():
    if address_count[letter] > maximum:
        letter_maximum = letter

print(letter)
```

*** =sample_code
```{python}
# write your code here!



```

*** =sct
```{python}
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
# Documentation can also be found at github.com/datacamp/pythonwhat/wiki
test_object("letter",
              not_called_msg = "Make sure to define `letter`!",
              incorrect_msg = "Check your usage of `letter` again.")
success_msg("Great work!")
```



