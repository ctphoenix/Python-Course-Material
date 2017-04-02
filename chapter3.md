---
title       : Case Study 1 - Caesar Cipher
description : A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of this message to another letter in the alphabet, which is a fixed number of letters away from the original.  If our encryption key were 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

*** =instructions
-  Create a dictionary `letters` with keys consisting of the numbers from 0 to 26, and values consisting of the lowercase letters of the English alphabet, including the space `' '` at the end.

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
-  `alphabet` and `letters` are already defined. Create a dictionary `encoding` with keys being the characters in `alphabet` and values being numbers from 0-26, shifted by an integer `encryption_key=3`.  For example, the key `a` should have value `encryption_key`, key `b` should have value `encryption_key + 1`, and so on.  If any result of this addition is less than 0 or greater than 26, you can ensure the result remains within 0-26 using `result % 27`.


*** =hint
- You can simply add `encryption_key` to the place value of each letter.  To reduce this `sum` by `27` if it exceeds `27`, use `sum % 27`!

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

encryption_key = 3

# define `encoding` here!


```

*** =solution
```{python}
alphabet = string.ascii_lowercase + " "
letters = dict(enumerate(alphabet))

encryption_key = 3

encoding = {letter: (place + encryption_key) % 27 for (place, letter) in enumerate(alphabet)}

```

*** =sct
```{python}
test_object("encoding",
            undefined_msg = "Did you define `encoding`?",
            incorrect_msg = "It looks like `encoding` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:95e2d3c0a4
## Exercise 3

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- `alphabet` and `letters` are preloaded from the previous exercise.  Write a function `caesar(message, encryption_key)` to encode a message with the Caesar cipher.
- Use your code from Exercise 2 to find the value of `encoding` for each letter in `message`.
- Use these values as keys in the dictionary `letters` to determine the encoded letter for each letter in `message`.
- Your function should return a string consisting of these encoded letters.
- Use `caesar` to encode `message` using `encryption_key = 3`, and save the result as `encoded_message`.
- Print `encoded_message`.

*** =hint
- Try using `"".join(my_list)` to transform the coded list into a string!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = string.ascii_lowercase + " "
message = "hi my name is caesar"
letters = dict(enumerate(alphabet))
encryption_key = 3
encoding = {letter: (place + encryption_key) % 27 for (place, letter) in enumerate(alphabet)}
```

*** =sample_code
```{python}
message = "hi my name is caesar"

def caesar(message, encryption_key):
    # return the encoded message as a single string!


```

*** =solution
```{python}
def caesar(message, encryption_key):
    """
    This is a Caesar cipher.  Each letter in a message is shifted by a few
    characters in the alphabet, and returned.

    message:        A string you would like to encode or decode.  Must consist of
                    lowercase letters and spaces.
    encryption_key: An integer, indicating how many characters each letter in the
                    message will be shifted.
    """
    encoding = {letter: (place + encryption_key) % 27 for (place, letter) in enumerate(alphabet)}
    coded_message = "".join([letters[encoding[letter]] for letter in message])
    return coded_message
    
encoded_message = caesar(message, encryption_key=3)
print(encoded_message)    
```

*** =sct
```{python}
test_student_typed("caesar",
              pattern=False,
              not_typed_msg="Make sure to call the function `caesar`!") 
test_student_typed("print",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")                
test_object("encoded_message",
            undefined_msg = "Did you define `encoded_message`?",
            incorrect_msg = "It looks like `encoded_message` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:99f93f9512
## Exercise 4

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

*** =instructions
- Note that `encoded_message` is already loaded from the previous problem. Use `caesar` to decode `encoded_message` using `encryption_key = -3`.
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
encryption_key = 3
encoding = {letter: (place + encryption_key) % 27 for (place, letter) in enumerate(alphabet)}
def caesar(message, encryption_key):
    encoding = {letter: (place + encryption_key) % 27 for (place, letter) in enumerate(alphabet)}
    return "".join([letters[encoding[letter]] for letter in message])
encoded_message = caesar(message, encryption_key=3)
```

*** =sample_code
```{python}



```

*** =solution
```{python}
decoded_message = caesar(encoded_message, encryption_key=-3)
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
success_msg("Great work!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```

