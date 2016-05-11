---
title       : Module 1 Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

A cipher is a secret code for a language.  For these bonus exercises, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The caesar cipher shifts each letter of this message to another letter in the alphabet, which is a fixed number of letters away from the original.  If our key was 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except move the same number of steps backwards in the alphabet.

We will perform this by converting letters to a number `0-26`, adding the key to each number, and looking up the letter corresponding to that number. To undo our cipher, we simply perform the same steps number (adding or subtracting `27` making sure each number stays within `0-26`).

*** =instructions
-  Create a dictionary "letters" that maps each letter in `alphabet` to the numbers `0-26`.

*** =hint
- `dict` and `enumerate` could come in handy!

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# Let's look at the lowercase letters.
import string
string.ascii_lowercase

# We will consider the alphabet to be these letters, along with a spaces.
alphabet = string.ascii_lowercase + " "

letters = dict(enumerate(alphabet))

```

*** =solution
```{python}
import string
string.ascii_lowercase
alphabet = string.ascii_lowercase + " "
letters = dict(enumerate(alphabet))
```

*** =sct
```{python}
#test_function("create_board",
#              not_called_msg = "Make sure to call `create_board`!",
#              incorrect_msg = "Check your definition of `create_board` again.")
test_object("letters",
            undefined_msg = "Did you define `letters`?",
            incorrect_msg = "It looks like `letters` wasn't defined correctly.")
success_msg("Great work!")
```

