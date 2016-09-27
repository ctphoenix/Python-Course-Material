---
title       : Module 1 (Translation) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

A cipher is a secret code for a language.  For these bonus exercises, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of this message to another letter in the alphabet, which is a fixed number of letters away from the original.  If our key was 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except move the same number of steps backwards in the alphabet.

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



--- type:NormalExercise lang:python xp:100 skills:1 key:2288897c84
## Exercise 2

A cipher is a secret code for a language.  For these bonus exercises, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
-  Create a dictionary `coded_letters` that rs (and the space) for keys, and numbers from `0-26` for values.  Define these alphabetically, starting with `a` mapped to  `key=3`.

*** =hint
- You can simply add `key` to the place value of each letter.  To reduce this `sum` by `27` if it exceeds `27`, use `sum%27`!

*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))

key = 3
# define `coded_letters` here!
```

*** =solution
```{python}
key = 3
coded_letters   = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}

```

*** =sct
```{python}
test_object("coded_letters",
            undefined_msg = "Did you define `coded_letters`?",
            incorrect_msg = "It looks like `coded_letters` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:95e2d3c0a4
## Exercise 3

A cipher is a secret code for a language.  For these bonus exercises, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- Create a function `caesar(message, key)` that includes `coded_letters` to encode a message with the caesar cipher.
- Encode, save and print the message defined earlier as `coded_message`, using `key = 3`.

*** =hint
- Using the code in Exercise 2, you can use `"".join(my_list)` to transform the coded list into a string!

*** =pre_exercise_code
```{python}
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))
key = 3
coded_letters   = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}
```

*** =sample_code
```{python}
def caesar(message, key):
    coded_letters = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}
    return "".join([letters[coded_letters[letter]] for letter in message])
coded_message = caesar(message, key=3)
print(coded_message)    
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
    coded_letters = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}
    return "".join([letters[coded_letters[letter]] for letter in message])
coded_message = caesar(message, key=3)
print(coded_message)    
```

*** =sct
```{python}
test_function("caesar",
              not_called_msg = "Make sure to call `caesar`!",
              incorrect_msg = "Check your definition of `caesar` again.")
test_function("print",
              not_called_msg = "Make sure to call `print`!",
              incorrect_msg = "Check your definition of `print` again.")
test_object("coded_message",
            undefined_msg = "Did you define `coded_message`?",
            incorrect_msg = "It looks like `coded_message` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:99f93f9512
## Exercise 4

A cipher is a secret code for a language.  For these bonus exercises, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- Decode, save and print your encoded message as `decoded_message`, using `key = -3`.  Does this recover your original message?

*** =hint
- This should not require any changes to the function `caesar`!

*** =pre_exercise_code
```{python}
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))
key = 3
coded_letters   = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}
def caesar(message, key):
    coded_letters = {letter:(place + key)%27 for (place, letter) in enumerate(alphabet)}
    return "".join([letters[coded_letters[letter]] for letter in message])
coded_message = caesar(message, key=3)
```

*** =sample_code
```{python}
decoded_message = caesar(coded_message, key=-3)
print(decoded_message)
```

*** =solution
```{python}
decoded_message = caesar(coded_message, key=-3)
print(decoded_message)
```

*** =sct
```{python}
test_function("print",
              not_called_msg = "Make sure to call `print`!",
              incorrect_msg = "Check your definition of `decoded_message` again.")
test_object("letters",
            undefined_msg = "Did you define `decoded_message`?",
            incorrect_msg = "It looks like `decoded_message` wasn't defined correctly.")
success_msg("Great work!")
```

