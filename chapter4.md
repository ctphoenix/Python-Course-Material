---
title       : Module 2 (Language Processing) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

For these exercises, functions `count_words_fast`, `read_book`, and `word_stats` are already defined from the main module.

*** =instructions
-  Create a function `word_count_distribution(text)` that takes a book (string) and outputs a dictionary with items corresponding to the count of times a collection of words appears in the translation, and values corresponding to the number of number of words that appear with that frequency.  Can you accomplish this by using `count_words_fast(text)` in your function?
- "Romeo and Juliet" is preloaded as `text`.  Call `word_count_distribution(text)`, and save the result as `distribution`.


*** =hint
- Using `count_words_fast(text)` and the values from `Counter` ought to do the trick!

*** =pre_exercise_code
```{python}
import os
import pandas as pd
import numpy as np
from collections import Counter
def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts
def read_book(title_path):
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")    
```

*** =sample_code
```{python}
# input your code here!

```

*** =solution
```{python}
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution

distribution = word_count_distribution(text) 
```

*** =sct
```{python}
test_function("word_count_distribution",
              not_called_msg = "Make sure to call `word_count_distribution`!",
              incorrect_msg = "Check your definition of `word_count_distribution` again.")              
test_object("distribution",
            undefined_msg = "Did you define `distribution`?",
            incorrect_msg = "It looks like `distribution` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:0fc5cd1ce9
## Exercise 2

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

For these exercises, functions `count_words_fast`, `read_book`, and `word_stats` are already defined from the main module.

*** =instructions
-  Edit the code used to read though each of the books in our library, and store each the word frequency distribution for each translation of William Shakespeare's "Hamlet" as a Pandas dataframe `hamlet`.  How many translations are there?  Which languages are they translated into?

*** =hint
- Define `hamlets` with columns `language` and `distribution`.  Then, add the results from `word_count_distribution(text)` as a row for all books with the title "Hamlet".

*** =pre_exercise_code
```{python}
import os
import pandas as pd
import numpy as np
from collections import Counter
def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts
def read_book(title_path):
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")    
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution   
```

*** =sample_code
```{python}
stats = pd.DataFrame(columns = ("language", "author", "title", "length", "unique"))
hamlets = pd.DataFrame(columns = ("language", "distribution"))
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    if language[0] != ".":
        for author in os.listdir(book_dir + "/" + language):
            if author[0] != ".":
                for title in os.listdir(book_dir + "/" + language + "/" + author):
                  #if title == "Hamlet.txt": #INCLUDE!
                    title = title.replace('"', "'")
                    inputfile = book_dir + "/" + language + "/" + author + "/" + title
                    print(inputfile)
                    text = read_book(inputfile)
                    (num_unique, counts) = word_stats(count_words(text)) #DELETE!
                    stats.loc[title_num] = language, author.title(), title.replace(".txt", "").title(), sum(counts), num_unique #DELETE!
                    #frequencies = word_count_distribution(text) # INCLUDE!
                    #hamlets.loc[title_num] = language, frequencies # INCLUDE!
                    title_num += 1
```

*** =solution
```{python}
hamlets = pd.DataFrame(columns = ("language", "distribution"))
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    if language[0] != ".":
        for author in os.listdir(book_dir + "/" + language):
            if author[0] != ".":
                for title in os.listdir(book_dir + "/" + language + "/" + author):
                    if title == "Hamlet.txt":
                        inputfile = book_dir + "/" + language + "/" + author + "/" + title
                        print(inputfile)
                        text = read_book(inputfile)
                        frequencies = word_count_distribution(text)
                        hamlets.loc[title_num] = language, frequencies
                        title_num += 1  
```

*** =sct
```{python}
test_object("hamlets",
            undefined_msg = "Did you define `hamlets`?",
            incorrect_msg = "It looks like `hamlets` wasn't defined correctly.")
success_msg("Great work!  There are three translations: English, German, and Portuguese.")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:f2cef742ec
## Exercise 3

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

For these exercises, functions `count_words_fast`, `read_book`, and `word_stats` are already defined from the main module.

*** =instructions
-  Create a function `more_frequent(distribution)` that takes a word frequency dict (like that made in Exercise 2) and outputs a dict with the same keys as those in distribution (the number of times a group of words appears) in the text), and values corresponding to the fraction of words that occur with more frequency than that.
-  Call `more_frequent(distribution)`.

*** =hint
- You might begin with sorting the counts of the distribution as follows: `counts = sorted(distribution.keys())`
- Sorting the values of the distribution with `sorted(distribution.values(), reverse = True)` and finding the cumulative sum of these using `np.cumsum()` will get you close!
- To obtain the fraction of words more frequent than this, divide this cumulative sum by its maximum, and subtract 1 from this value.  You're ready to make a dictionary with these as values and counts as keys!

def more_frequent(distribution):
    counts = sorted(distribution.keys())
    sorted_frequencies = sorted(distribution.values(), reverse = True)
    cumulative_frequencies = np.cumsum(sorted_frequencies)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))
    
more_frequent(distribution)

*** =pre_exercise_code
```{python}
import os
import pandas as pd
import numpy as np
from collections import Counter
def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts
def read_book(title_path):
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution    
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")    
hamlets = pd.DataFrame(columns = ("language", "distribution"))
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    if language[0] != ".":
        for author in os.listdir(book_dir + "/" + language):
            if author[0] != ".":
                for title in os.listdir(book_dir + "/" + language + "/" + author):
                    if title == "Hamlet.txt":
                        inputfile = book_dir + "/" + language + "/" + author + "/" + title
                        text = read_book(inputfile)
                        frequencies = word_count_distribution(text)
                        hamlets.loc[title_num] = language, frequencies
                        title_num += 1         
```

*** =sample_code
```{python}
# input your code here!

```

*** =solution
```{python}
def more_frequent(distribution):
    counts = sorted(distribution.keys())
    sorted_frequencies = sorted(distribution.values(), reverse = True)
    cumulative_frequencies = np.cumsum(sorted_frequencies)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))
    
more_frequent(distribution)    
```

*** =sct
```{python}
test_function("more_frequent",
              not_called_msg = "Make sure to call `more_frequent`!",
              incorrect_msg = "Check your definition of `more_frequent` again.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:62f73c5919
## Exercise 4

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

For these exercises, functions `count_words_fast`, `read_book`, and `word_stats` are already defined from the main module.

*** =instructions
-  Plot the word frequency distributions of each translations on a single log-log plot.  Make sure to include a legend and a title!  Do the distributions differ?

*** =hint
- No hint for this one!

*** =pre_exercise_code
```{python}
import os
import pandas as pd
import numpy as np
from collections import Counter
def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts
def read_book(title_path):
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution     
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")    
hamlets = pd.DataFrame(columns = ("language", "distribution"))
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    if language[0] != ".":
        for author in os.listdir(book_dir + "/" + language):
            if author[0] != ".":
                for title in os.listdir(book_dir + "/" + language + "/" + author):
                    if title == "Hamlet.txt":
                        inputfile = book_dir + "/" + language + "/" + author + "/" + title
                        text = read_book(inputfile)
                        frequencies = word_count_distribution(text)
                        hamlets.loc[title_num] = language, frequencies
                        title_num += 1         
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import numpy as np
```

*** =sample_code
```{python}
colors = ["crimson", "forestgreen", "blueviolet"]
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]
    # call `more_frequent` with input `distribution`, and store as `dist`.
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2)
    handles.append(plot)
    hamlet_languages.append(language)
# call `plt.title` here, with a string argument for a title.
xlabel  = "Word Frequency"
ylabel  = "Probability of Words Being More Frequent"
plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
plt.show()
```

*** =solution
```{python}
colors = ["crimson", "forestgreen", "blueviolet"]
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]# frequencies is count_distribution
    dist = more_frequent(distribution)
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2)
    handles.append(plot)
    hamlet_languages.append(language)
plt.title("Distributions of Word Frequencies in Hamlet Translations")
xlabel  = "Word Frequency"
ylabel  = "Probability of Words Being More Frequent"
plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
# show your plot using `plt.show`!
```

*** =sct
```{python}
test_function("plt.show",
              not_called_msg = "Make sure to show your plot using `plt.show`!")
test_function("plt.legend",
              not_called_msg = "Make sure to include a legend using `plt.legend`!")     
test_function("plt.title",
              not_called_msg = "Make sure to include a .title using `plt..title`!")     
success_msg("Great work!  The distributions differ somewhat, but their basic shape is the same.  By the way, distributions that look like a straight line like these are called "scale-free," because the line looks the same no matter where on the x-axis you look!")
```


