---
title       : Case Study 2 - Translations of Hamlet
description : In this case study, we will find and plot the distribution of word frequencies for each translation of Hamlet.  Perhaps the distribution of word frequencies of Hamlet depends on the translation - let's find out!
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341
## Exercise 1

In this case study, we will find and visualize summary statistics of the text of different translations of Hamlet. For this case study, functions `count_words_fast`, `read_book`, and `word_stats` are already defined as in the Case 2 Videos (Videos 3.2.x).

In this exercise, we will first read in and store each translation of Hamlet. 

*** =instructions
- Define `hamlets` as a `pandas` dataframe with columns `language` and `text`.
- Add an `if` statement to check if the title `'Hamlet'`.
- Store the results from `read_book(inputfile)` to `text`.
- Consider: How many translations are there? Which languages are they translated into?

*** =hint
- Try using `pd.DataFrame`.  Make sure to include the `columns` argument!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
book_titles = {#only a selection for now, as the exercises only require translations of Hamlet.
    "English": {
        "shakespeare": ("A+Midsummer+Night's+Dream", "Hamlet", "Macbeth", "Othello", "Richard+III", "Romeo+and+Juliet", "The+Merchant+of+Venice")
    },
    "French": {
        "chevalier":     ("L'enfer+et+le+paradis+de+l'autre+monde", "L'i%CC%82le+de+sable", "La+capitaine","La+fille+des+indiens+rouges", "La+fille+du+pirate", "Le+chasseur+noir", "Les+derniers+Iroquois")
    },
    "German": {
        "shakespeare":   ("Der+Kaufmann+von+Venedig", "Ein+Sommernachtstraum", "Hamlet", "Macbeth", "Othello", "Richard+III", "Romeo+und+Julia")
    },
    "Portuguese": {
        "shakespeare":   ("Hamlet", )
    }
}
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
    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
    text = text.to_string(index = False)
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution
def more_frequent(distribution):
    counts = list(distribution.keys())
    frequency_of_counts = list(distribution.values())
    cumulative_frequencies = np.cumsum(frequency_of_counts)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))
```

*** =sample_code
```{python}
hamlets = ## Enter code here! ##
book_dir = "Books"
title_num = 1
for language in book_titles:
    for author in book_titles[language]:
        for title in book_titles[language][author]:
            if ## Enter code here! ##
                inputfile = data_filepath+"Books/"+language+"/"+author+"/"+title+".txt"
                text = ## Enter code here! ##
                hamlets.loc[title_num] = language, text
                title_num += 1

                
```

*** =solution
```{python}
hamlets = pd.DataFrame(columns = ["language","text"])
book_dir = "Books"
title_num = 1
for language in book_titles:
    for author in book_titles[language]:
        for title in book_titles[language][author]:
            if title == "Hamlet":
                inputfile = data_filepath+"Books/"+language+"/"+author+"/"+title+".txt"
                text = read_book(inputfile)
                hamlets.loc[title_num] = language, text
                title_num += 1
```

*** =sct
```{python}
test_object("hamlets",
            undefined_msg = "Did you define `hamlets`?",
            incorrect_msg = "It looks like `hamlets` wasn't defined correctly.")
success_msg("Great work!  There are three translations: English, German, and Portuguese.")
```


--- type:NormalExercise lang:python xp:100 skills:2 key:f2cef742ec
## Exercise 2

In this case study, we will find and visualize summary statistics of the text of different translations of Hamlet. For this case study, functions `count_words_fast`, `read_book`, and `word_stats` are already defined as in the Case 2 Videos (Videos 3.2.x)

In this exercise, we will summarize the text for a single translation of Hamlet in a `pandas` dataframe. The language and text of the first translation of Hamlet in `hamlets` is given in the code section.

*** =instructions
- Find the dictionary of word frequency in `text` by calling `count_words_fast()`. Store this as `counted_text`.
- Create a `pandas` dataframe named `data`.
- Using ``counted_text`, define two columns in `data`:
 -  `word`, consisting of each unique word in `text`.
 -  `count`, consisting of the number of times each word in `word` is included in the text.

*** =hint
- `word` are the keys and `count` are the values from `counted_text`.
- They may be included by converting `counted_text.keys()` and `counted_text.values()` into lists.

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
book_titles = {#only a selection for now, as the exercises only require translations of Hamlet.
    "English": {
        "shakespeare": ("A Midsummer Night's Dream", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo and Juliet", "The Merchant of Venice")
    },
    "French": {
        "chevalier":     ("L'enfer et le paradis de l'autre monde", "L'i%CC%82le de sable", "La capitaine","La fille des indiens rouges", "La fille du pirate", "Le chasseur noir", "Les derniers Iroquois")
    },
    "German": {
        "shakespeare":   ("Der Kaufmann von Venedig", "Ein Sommernachtstraum", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo und Julia")
    },
    "Portuguese": {
        "shakespeare":   ("Hamlet", )
    }
}
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
    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
    text = text.to_string(index = False)
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution
text = read_book(data_filepath + "Books/English/shakespeare/Romeo+and+Juliet.txt") 
distribution = word_count_distribution(text) 

hamlets = pd.DataFrame(columns = ["language","text"])
book_dir = "Books"
title_num = 1
for language in book_titles:
    for author in book_titles[language]:
        for title in book_titles[language][author]:
            if title == "Hamlet":
                inputfile = data_filepath+"Books/"+language+"/"+author+"/"+title+".txt"
                text = read_book(inputfile)
                hamlets.loc[title_num] = [language, text]
                title_num += 1
```

*** =sample_code
```{python}
language, text = hamlets.iloc[0]

# Enter your code here.


```

*** =solution
```{python}
language, text = hamlets.iloc[0]

counted_text = count_words_fast(text)

data = pd.DataFrame({
    "word": list(counted_text.keys()),
    "count": list(counted_text.values())
})
```

*** =sct
```{python}
test_function("count_words_fast",
              not_called_msg = "Make sure to call `count_words_fast`!",
              incorrect_msg = "Check your definition of `count_words_fast` again.")
test_object("data",
            undefined_msg = "Did you define `data`?",
            incorrect_msg = "It looks like `data` wasn't defined correctly.")
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:0fc5cd1ce9
## Exercise 3

In this case study, we will find and visualize summary statistics of the text of different translations of Hamlet. For this case study, functions `count_words_fast`, `read_book`, and `word_stats` are already defined as in the Case 2 Videos (Videos 3.2.x)

In this exercise, we will continue to define summary statistics for a single translation of Hamlet. The solution code from the previous section is already included here.

*** =instructions
- Add a column to `data` named `length`, defined as the length of each word.
- Add another column named `frequency`, which is defined as follows for each word in `data`:
 -  If `count` > 10, `frequency ` is `frequent`.
 -  If 1 < `count` <= 10, `frequency ` is `infrequent`.
 -  If `count` == 1, `frequency ` is `unique`.

*** =hint
- You can use the `apply()` function to `data["word"]` to apply a function to each element in `data["word"]`. To compute the length of each word, try the `len` function as the argument for `apply()`.
- The column `frequency` can be defined using cases. `data.loc[]` can be used to locate the rows that meet a certain criterion, such as `data["count"] > 10`.  

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
book_titles = {#only a selection for now, as the exercises only require translations of Hamlet.
    "English": {
        "shakespeare": ("A Midsummer Night's Dream", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo and Juliet", "The Merchant of Venice")
    },
    "French": {
        "chevalier":     ("L'enfer et le paradis de l'autre monde", "L'i%CC%82le de sable", "La capitaine","La fille des indiens rouges", "La fille du pirate", "Le chasseur noir", "Les derniers Iroquois")
    },
    "German": {
        "shakespeare":   ("Der Kaufmann von Venedig", "Ein Sommernachtstraum", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo und Julia")
    },
    "Portuguese": {
        "shakespeare":   ("Hamlet", )
    }
}
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
    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
    text = text.to_string(index = False)
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution
text = read_book(data_filepath + "Books/English/shakespeare/Romeo+and+Juliet.txt") 
distribution = word_count_distribution(text) 

hamlets = pd.DataFrame(columns = ["language","text"])
book_dir = "Books"
title_num = 1
for language in book_titles:
    for author in book_titles[language]:
        for title in book_titles[language][author]:
            if title == "Hamlet":
                inputfile = data_filepath+"Books/"+language+"/"+author+"/"+title+".txt"
                text = read_book(inputfile)
                hamlets.loc[title_num] = [language, text]
                title_num += 1
                
```

*** =sample_code
```{python}
language, text = hamlets.iloc[0]

counted_text = count_words_fast(text)

data = pd.DataFrame({
    "word": list(counted_text.keys()),
    "count": list(counted_text.values())
})

# Enter your code here.


```

*** =solution
```{python}
language, text = hamlets.iloc[0]

counted_text = count_words_fast(text)

data = pd.DataFrame({
    "word": list(counted_text.keys()),
    "count": list(counted_text.values())
})

data["length"] = data["word"].apply(len)

data.loc[data["count"] > 10,  "frequency"] = "frequent"
data.loc[data["count"] <= 10, "frequency"] = "infrequent"
data.loc[data["count"] == 1,  "frequency"] = "unique"
```

*** =sct
```{python}
test_object("data",
            undefined_msg = "Did you define `data`?",
            incorrect_msg = "It looks like `data` wasn't defined correctly.")
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:62f73c5919
## Exercise 4

In this case study, we will find and visualize summary statistics of the text of different translations of Hamlet.

For these exercises, functions `count_words_fast`, `read_book`, and `word_stats` are already defined as in the Case 2 Videos (Videos 3.2.x)

*** =instructions
-  Plot the word frequency distributions of each translations on a single log-log plot.  Note that we have already done most of the work for you.  Do the distributions of each translation differ?

*** =hint
- No hint for this one: don't overthink it!

*** =pre_exercise_code
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
book_titles = {#only a selection for now, as the exercises only require translations of Hamlet.
    "English": {
        "shakespeare": ("A Midsummer Night's Dream", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo and Juliet", "The Merchant of Venice")
    },
    "French": {
        "chevalier":     ("L'enfer et le paradis de l'autre monde", "L'i%CC%82le de sable", "La capitaine","La fille des indiens rouges", "La fille du pirate", "Le chasseur noir", "Les derniers Iroquois")
    },
    "German": {
        "shakespeare":   ("Der Kaufmann von Venedig", "Ein Sommernachtstraum", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo und Julia")
    },
    "Portuguese": {
        "shakespeare":   ("Hamlet", )
    }
}
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
    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
    text = text.to_string(index = False)
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
def word_count_distribution(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution
def more_frequent(distribution):
    counts = list(distribution.keys())
    frequency_of_counts = list(distribution.values())
    cumulative_frequencies = np.cumsum(frequency_of_counts)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))
hamlets = pd.DataFrame(columns = ("language", "distribution"))
book_dir = "Books"
title_num = 1
for language in book_titles:
    for author in book_titles[language]:
        for title in book_titles[language][author]:
            if title == "Hamlet":
                inputfile = data_filepath+"Books/"+language+"/"+author+"/"+title+".txt"
                inputfile.replace(" ","+")
                text = read_book(inputfile)
                distribution = word_count_distribution(text)
                hamlets.loc[title_num] = language, distribution
                title_num += 1
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict

```

*** =sample_code
```{python}
colors = ["crimson", "forestgreen", "blueviolet"]
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]
    dist = more_frequent(distribution)
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2)
    handles.append(plot)
    hamlet_languages.append(language)
plt.title("Word Frequencies in Hamlet Translations")
xlim    = [0, 2e3]
xlabel  = "Frequency of Word $W$"
ylabel  = "Fraction of Words\nWith Greater Frequency than $W$"
plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
# show your plot using `plt.show`!


```

*** =solution
```{python}
colors = ["crimson", "forestgreen", "blueviolet"]
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]
    dist = more_frequent(distribution)
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2)
    handles.append(plot)
    hamlet_languages.append(language)
plt.title("Word Frequencies in Hamlet Translations")
xlim    = [0, 2e3]
xlabel  = "Frequency of Word $W$"
ylabel  = "Fraction of Words\nWith Greater Frequency than $W$"
plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
plt.show()
```

*** =sct
```{python}
test_student_typed("plt.show()",
              pattern=False,
              not_typed_msg="Did you use `plt.show`?")   
success_msg("Great work!  The distributions differ somewhat, but their basic shape is the same.  By the way, distributions that look like a straight line like these are called 'scale-free,' because the line looks the same no matter where on the x-axis you look!  This concludes the case study.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+3T2016")
```



--- type:NormalExercise lang:python xp:100 skills:2 key:c3d2e7f96f
## Exercise 5


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}

```

*** =solution
```{python}

```

*** =sct
```{python}

```
