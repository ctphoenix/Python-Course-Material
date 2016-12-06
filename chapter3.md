---
title       : Case Study 1 - Caesar Cipher
description : A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of this message to another letter in the alphabet, which is a fixed number of letters away from the original.  If our key were 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

*** =instructions
-  Create a dictionary `letters` that maps each number from 0 to 26 to each character in `alphabet`.

*** =hint
- `dict` and `enumerate` could come in handy!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
# Let's look at the lowercase letters.
import string
string.ascii_lowercase

# We will consider the alphabet to be these letters, along with a space.
alphabet = string.ascii_lowercase + " "

# create `letters` here!

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
test_object("letters",
            undefined_msg = "Did you define `letters`?",
            incorrect_msg = "It looks like `letters` wasn't defined correctly.")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:2288897c84
## Exercise 2

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
-  `alphabet` and `letters` are shown below.    Create a dictionary `coded_message` with keys being the characters in `alphabet`, and values being numbers from `0-26` shifted by an integer `key`.  Define these alphabetically, starting with `a` mapped to `key=3`.

*** =hint
- You can simply add `key` to the place value of each letter.  To reduce this `sum` by `27` if it exceeds `27`, use `sum % 27`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_lowercase + " "
letters = dict(enumerate(alphabet))
```

*** =sample_code
```{python}
alphabet = string.ascii_lowercase + " "
letters = dict(enumerate(alphabet))

key = 3

# define `coded_message` here!


```

*** =solution
```{python}
alphabet = string.ascii_lowercase + " "
letters = dict(enumerate(alphabet))

key = 3

coded_message = {letter: (place + key) % 27 for (place, letter) in enumerate(alphabet)}

```

*** =sct
```{python}
test_object("coded_message",
            undefined_msg = "Did you define `coded_message`?",
            incorrect_msg = "It looks like `coded_message` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:95e2d3c0a4
## Exercise 3

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- Create a function `caesar(message, key)` that includes `coded_message` to encode a message with the caesar cipher.  `alphabet` and `letters` are preloaded from the previous exercise.
- Use `caesar` to encode `message` using `key = 3`, and save the result as `coded_message`.
- Print `coded_message`.

*** =hint
- Try using `"".join(my_list)` to transform the coded list into a string!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))
key = 3
coded_message = {letter: (place + key) % 27 for (place, letter) in enumerate(alphabet)}
```

*** =sample_code
```{python}
message = "hi my name is caesar"

def caesar(message, key):
    # return the encoded message as a single string!


```

*** =solution
```{python}
def caesar(message, key):
    """
    This is a Caesar cipher.  Each letter in a message is shifted by a few
    characters in the alphabet, and returned.

    message: A string you would like to encode or decode.  Must consist of
             lowercase letters and spaces.
    key:     An integer, indicating how many characters each letter in the
             message will be shifted.
    """
    coded_message = {letter: (place + key) % 27 for (place, letter) in enumerate(alphabet)}
    coded_message_string = "".join([letters[coded_message[letter]] for letter in message])
    return coded_message_string
    
coded_message = caesar(message, key=3)
print(coded_message)    
```

*** =sct
```{python}
test_function("caesar",
              not_called_msg = "Make sure to call `caesar`!",
              incorrect_msg = "Check your definition of `caesar` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")                
test_object("coded_message",
            undefined_msg = "Did you define `coded_message`?",
            incorrect_msg = "It looks like `coded_message` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:99f93f9512
## Exercise 4

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- Decode and save `coded_message` using `caesar` and `key = -3`.  `coded_message` is already loaded from the previous problem.
- Store your decoded message as `decoded_message`.
- Print `decoded_message`.  Does this recover your original message?

*** =hint
- This should not require any changes to the function `caesar`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))
key = 3
coded_message = {letter: (place + key) % 27 for (place, letter) in enumerate(alphabet)}
def caesar(message, key):
    coded_message = {letter: (place + key) % 27 for (place, letter) in enumerate(alphabet)}
    return "".join([letters[coded_message[letter]] for letter in message])
coded_message = caesar(message, key=3)
```

*** =sample_code
```{python}



```

*** =solution
```{python}
decoded_message = caesar(coded_message, key=-3)
print(decoded_message)
```

*** =sct
```{python}
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")  
test_object("decoded_message",
            undefined_msg = "Did you define `decoded_message`?",
            incorrect_msg = "It looks like `decoded_message` wasn't defined correctly.")
success_msg("Great work!")
```

