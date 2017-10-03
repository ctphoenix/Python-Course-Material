---
title       : Case Study 1 - Caesar Cipher
description : A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

## Exercise 1

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of a message to another letter in the alphabet located a fixed distance from the original letter. If our encryption key were 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

Over the next five exercises, we will create our own Caesar cipher, as well as a message decoder for this cipher. In this exercise, we will define the alphabet used in the cipher.

*** =instructions
-  The `string` library has been imported. Create a string called `alphabet` consisting of the lowercase letters of the space character space `' '`, concatenated with `string.ascii_lowercase` at the end.

*** =hint

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
```

*** =sample_code
```{python}
# Let's look at the lowercase letters.
import string
alphabet = # Add your code here! #

```

*** =solution
```{python}
import string
alphabet = " " + string.ascii_lowercase
```

*** =sct
```{python}
test_object("alphabet",
            undefined_msg = "Did you define `alphabet`?",
            incorrect_msg = "It looks like `alphabet` wasn't defined correctly.")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:2288897c84
## Exercise 2

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of a message to another letter in the alphabet located a fixed distance from the original letter. If our encryption key were 1, we would shift `h` to the next letter `i`, `i` to the next letter `j`, and so on.  If we reach the end of the alphabet, which for us is the space character, we simply loop back to `a`. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

In this exercise, we will define a dictionary that specifies the index of each character in `alphabet`.

*** =instructions
-  `alphabet` has already defined from the last exercise. Create a dictionary `letters` with keys consisting of the characters in `alphabet`, and values consisting of the numbers from 0 to 26.
- Store this as `positions`.

*** =hint
- You can use a `for` loop to iterate through the values of `alphabet` and assign each as a key to `positions` with a value that increments by `1` after every new entry.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = " " + string.ascii_lowercase
```

*** =sample_code
```{python}
positions =

```

*** =solution
```{python}
alphabet = " " + string.ascii_lowercase
positions = {}
index = 0
for char in alphabet:
    positions[char] = index
    index += 1
```

*** =sct
```{python}
test_object("positions",
            undefined_msg = "Did you define `positions`?",
            incorrect_msg = "It looks like `positions` wasn't defined correctly.")
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:70b4ba58eb
## Exercise 3

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

In this exercise, we will encode a message with a Caesar cipher.

*** =instructions
-  `alphabet` and `positions` have already been defined from previous exercises. Use `positions` to create an encoded message based on `message` where each character in `message` has been shifted forward by 1 position, as defined by `positions`. Note that you can ensure the result remains within 0-26 using `result % 27`
- Store this as `encoded_message`.

*** =hint
- You might use a `for` loop that calls the position of each character in `message`, and increment this value by 1.
- Using these incremented positions as indices of `alphabet`, you will recover the correct encoded character of the message!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = " " + string.ascii_lowercase
positions = {}
index = 0
for char in alphabet:
    positions[char] = index
    index += 1
```

*** =sample_code
```{python}
message = "hi my name is caesar"


```

*** =solution
```{python}
message = "hi my name is caesar"

encoding_list = []
for char in message:
    position = positions[char]
    encoded_position = (position + 1) % 27
    encoding_list.append(alphabet[encoded_position])
encoded_message = "".join(encoding_list)
```

*** =sct
```{python}
test_object("encoded_message",
            undefined_msg = "Did you define `encoded_message`?",
            incorrect_msg = "It looks like `encoded_message` wasn't defined correctly.")
success_msg("Great work!")
```





--- type:NormalExercise lang:python xp:100 skills:2 key:95e2d3c0a4
## Exercise 4

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

In this exercise, we will define a function that encodes a message with any given encryption key.

*** =instructions
- `alphabet`, `position` and `message` remain defined from previous exercises. In addition, sample code for the previous exercise is provided below. Modify this code to define a function `encoding` that takes a message as input as well as an `int` encryption key `key` to encode a message with the Caesar cipher by shifting each letter in `message` by `key` positions.
- Your function should return a string consisting of these encoded letters.
- Use `encode` to encode `message` using `key = 3`, and save the result as `encoded_message`.
- Print `encoded_message`.

*** =hint
- Much of what is being asked of you is to transform the given code into a function, and call and print its use.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = " " + string.ascii_lowercase
message = "hi my name is caesar"
positions = {}
index = 0
for char in alphabet:
    positions[char] = index
    index += 1
```

*** =sample_code
```{python}
encoding_list = []
for char in message:
    position = positions[char]
    encoded_position = (position + key) % 27
    encoding_list.append(alphabet[encoded_position])
encoded_string = "".join(encoding_list)


```

*** =solution
```{python}
def encoding(message, key = 0):
    """
    This is a Caesar cipher.  Each letter in a message is shifted by a few
    characters in the alphabet, and returned.

    message:        A string you would like to encode or decode.  Must consist of
                    lowercase letters and spaces.
    key: An integer, indicating how many characters each letter in the
                    message will be shifted.
    """
    encoding_list = []
    for char in message:
        position = positions[char]
        encoded_position = (position + key) % 27
        encoding_list.append(alphabet[encoded_position])
    encoded_string = "".join(encoding_list)
    return encoded_string
    
encoded_message = encoding(message, 3)
print(encoded_message)

```

*** =sct
```{python}
test_student_typed("encoding(",
              pattern=False,
              not_typed_msg="Make sure to call the function `encoding`!") 
test_student_typed("print(",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")                
test_object("encoded_message",
            undefined_msg = "Did you define `encoded_message`?",
            incorrect_msg = "It looks like `encoded_message` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:99f93f9512
## Exercise 5

A cipher is a secret code for a language.  In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

In this exercise, we will decode an encoded message.

*** =instructions
- Note that `encoding` and `encoded_message` are already loaded from the previous problem. Use `encoding` to decode `encoded_message` using `key = -3`.
- Store your decoded message as `decoded_message`.
- Print `decoded_message`.  Does this recover your original message?

*** =hint
- This should not require any changes to the function `encoding`!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import string
alphabet = " " + string.ascii_lowercase
message = "hi my name is caesar"
positions = {}
index = 0
for char in alphabet:
    positions[char] = index
    index += 1

def encoding(message, key = 0):
    """
    This is a Caesar cipher.  Each letter in a message is shifted by a few
    characters in the alphabet, and returned.

    message:        A string you would like to encode or decode.  Must consist of
                    lowercase letters and spaces.
    key: An integer, indicating how many characters each letter in the
                    message will be shifted.
    """
    encoding_list = []
    for char in message:
        position = positions[char]
        encoded_position = (position + encode_key) % 27
        encoding_list.append(alphabet[encoded_position])
    encoded_string = "".join(encoding_list)
    return encoded_string

encoded_message = encoding(message, 3)
```

*** =sample_code
```{python}

decoded_message = 

# print your decoded message here!

```

*** =solution
```{python}
decoded_message = encoding(encoded_message, -3)

print(decoded_message)

```

*** =sct
```{python}
test_student_typed("print(",
              pattern=False,
              not_typed_msg="Make sure to call `print`!")  
test_object("decoded_message",
            undefined_msg = "Did you define `decoded_message`?",
            incorrect_msg = "It looks like `decoded_message` wasn't defined correctly.")
success_msg("Great work!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```

