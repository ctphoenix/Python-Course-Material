---
title       : Module 2 (Language Processing) Extra Exercises
description : Exercises for Homework
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

*** =instructions
-  

*** =hint
- 

*** =pre_exercise_code
```{python}
import os
import pandas as pd
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
```

*** =sample_code
```{python}

```

*** =solution
```{python}
```

*** =sct
```{python}
#test_function("print",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
#test_object("",
#            undefined_msg = "Did you define ``?",
#            incorrect_msg = "It looks like `` wasn't defined correctly.")
#success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

*** =instructions
-  

*** =hint
- 

*** =pre_exercise_code
```{python}
import os
import pandas as pd
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
```

*** =sample_code
```{python}

```

*** =solution
```{python}
```

*** =sct
```{python}
#test_function("print",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
#test_object("",
#            undefined_msg = "Did you define ``?",
#            incorrect_msg = "It looks like `` wasn't defined correctly.")
#success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:07ea54b341
## Exercise 1

In these bonus exercises, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

*** =instructions
-  

*** =hint
- 

*** =pre_exercise_code
```{python}
import os
import pandas as pd
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
```

*** =sample_code
```{python}

```

*** =solution
```{python}
```

*** =sct
```{python}
#test_function("print",
#              not_called_msg = "Make sure to call ``!",
#              incorrect_msg = "Check your definition of `` again.")
#test_object("",
#            undefined_msg = "Did you define ``?",
#            incorrect_msg = "It looks like `` wasn't defined correctly.")
#success_msg("Great work!")
```























# ############################# Bonus Exercises ############################# #

# In these bonus exercises, we will find and plot the distribution of word
# frequencies for each translation of Hamlet.  Perhaps the distribution of word
# frequencies of Hamlet depends on the translation --- let's find out!

##book_dir = "./Books"
##title_num = 1
##stats = pd.DataFrame(columns = ("language", "author", "title", "length", "unique"))
##for language in os.listdir(book_dir):
##    if language[0] != ".":
##        for author in os.listdir(book_dir + "/" + language):
##            if author[0] != ".":
##                for title in os.listdir(book_dir + "/" + language + "/" + author):
##                    #title = title.replace('"', "'") # change single quotes to double quotes
##                    inputfile = book_dir + "/" + language + "/" + author + "/" + title
##                    print(inputfile)
##                    text = read_book(inputfile)
##                    (num_unique, counts) = word_stats(count_words(text))
##                    stats.loc[title_num] = language, author.title(), title.replace(".txt", "").title(), sum(counts), num_unique
##                    title_num += 1


# Exercise 1: Create a function word_count_distribution(text) that takes a book
# (string) and outputs a dictionary with items corresponding to the count of
# times a collection of words appears in the translation, and values
# corresponding to the number of number of words that appear with that
# frequency.  Can you accomplish this by using count_words_fast(text) in your
# function?

def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution


# Exercise 2:  Edit the code used to read though each of the books in our
# library, and store each the word frequency distribution for each translation
# of William Shakespeare's "Hamlet" as a Pandas dataframe.  How many translations
# are there?  Which languages are they translated into?

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

# There are three translations: English, German, and Portuguese.

# Exercise 3: Create a function more_frequent(distribution) that takes a word
# frequency dict (like that made in Question 2) and outputs a dict with the same
# keys as those in distribution (the number of times a group of words appears)
# in the text), and values corresponding to the fraction of words that occur
# with more frequency than that.

def more_frequent(distribution):
    counts = sorted(distribution.keys())
    sorted_frequencies = sorted(distribution.values(), reverse = True)
    cumulative_frequencies = np.cumsum(sorted_frequencies)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))


# Exercise 4:  Plot the word frequency distributions of each translations on
# a single log-log plot.  Make sure to include a legend!  Do the distributions
# differ?

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import numpy as np

colors = ["crimson", "forestgreen", "blueviolet"]

plot_file = PdfPages("distributions.pdf")
fig = plt.figure()
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]# frequencies is count_distribution
    dist = more_frequent(distribution)
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2) #1
    handles.append(plot)
    hamlet_languages.append(language)
plt.title("Distributions of Word Frequencies in Hamlet Translations") #2
xlabel  = "Word Frequency" #3
ylabel  = "Probability of Words Being More Frequent" #3
plt.xlabel(xlabel); plt.ylabel(ylabel) #3
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1) #4
plot_file.savefig(fig)
plt.close()
plot_file.close()

# The distributions differ somewhat, but their basic shape is the same.  By the
# way, distributions that look like a straight line like these are called
# "scale-free," because the line looks the same no matter where on the x-axis
# you look!









