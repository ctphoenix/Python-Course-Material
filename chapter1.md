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






