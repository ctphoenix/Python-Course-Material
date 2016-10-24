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
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
text = read_book(data_filepath + "Books/English/shakespeare/Romeo+and+Juliet.txt") 
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
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
    counts = sorted(distribution.keys())
    sorted_frequencies = sorted(distribution.values(), reverse = True)
    cumulative_frequencies = np.cumsum(sorted_frequencies)
    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
    return dict(zip(counts, more_frequent))
```

*** =sample_code
```{python}
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
                frequencies = word_count_distribution(text)
                hamlets.loc[title_num] = language, frequencies
                title_num += 1
```

*** =solution
```{python}
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
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
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
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
book_titles = {
    "English": {
        "shakespeare": ("A Midsummer Night's Dream", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo and Juliet", "The Merchant of Venice")
    },
    "French": {
        "chevalier":     ("L'enfer et le paradis de l'autre monde", "L'i%CC%82le de sable", "La capitaine","La fille des indiens rouges", "La fille du pirate", "Le chasseur noir", "Les derniers Iroquois"),
        "de Maupassant": ("Boule de Suif", "Claire de Lune", "Contes de la Becasse", "L'inutile beaute%CC%81", "La Main Gauche", "La Maison Tellier", "La petite roque", "Le Horla", "%C5%92uvres comple%CC%80tes de Guy de Maupassant"),
        "diderot":       ("Ceci n'est pas un conte", "Entretien d'un pe%CC%80re avec ses enfants", "L'oiseau blanc", "Les deux amis de Bourbonne", "Regrets sur ma vieille robe de chambre"),
        "sand":          ("Jacques le fataliste et son mai%CC%82tre", "L' Orco", "La Coupe; Lupo Liverani; Le Toast; Garnier; Le Contrebandier; La Re%CC%82verie a%CC%80  Paris", "La Marquise", "Le Piccinino", "Le poe%CC%88me de Myrza", "Mattea", "Metella", "Oeuvres illustre%CC%81es de George Sand", "Pauline", "cora")
    },
    "German": {
        "lessing":       ("der freigeist", "der junge gelehrte", "die juden", "emilia galotti", "hamburgische dramaturgie", "minna von barnhelm", "miss sara sampson", "philotas"),
        "raimund":       ("Das Ma%CC%88dchen aus der Feenwelt", "Der Alpenko%CC%88nig und der Menschenfeind", "Der Barometermacher auf der Zauberinsel", "Der Diamant des Geisterko%CC%88nigs", "Der Verschwender", "Die gefesselte Phantasie", "Die unheilbringende Krone", "Moisasurs Zauberfluch"),
        "schiller":      ("Der Neffe als Onkel", "Der Parasit, oder die Kunst, sein Glu%CC%88ck zu machen",  "Die Jungfrau von Orleans", "Die Piccolomini", "Die Verschwo%CC%88rung des Fiesco zu Genua", "Kabale und Liebe", "Turandot, Prinzessin von China", "Wallensteins Lager", "Wallensteins Tod", "die braut von messina"),
        "shakespeare":   ("Der Kaufmann von Venedig", "Ein Sommernachtstraum", "Hamlet", "Macbeth", "Othello", "Richard III", "Romeo und Julia")
    },
    "Portuguese": {
        "Queiros":       ("A Cidade e as Serras", "A Illustre Casa de Ramires", "A Reli%CC%81quia", "A corresponde%CC%82ncia de Fradique Mendes", "Cartas de Inglaterra", "O Mandarim", "O Primo Bazilio", "O crime do padre Amaro", "Os Maias"),
        "branco":        ("A Filha do Arcediago", "A Neta do Arcediago", "A Queda d'um Anjo", "Agulha em Palheiro", "Amor de Perdic%CC%A7a%CC%83o", "Amor de Salvac%CC%A7a%CC%83o", "Annos de Prosa", "Carlota Angela", "Estrellas Funestas", "Estrellas Propi%CC%81cias", "Lagrimas Abenc%CC%A7oadas", "Livro de Consolac%CC%A7a%CC%83o", "O Olho de Vidro", "O Regicida", "O que fazem mulheres", "Scenas Contemporaneas"),
        "dinis":         ("A Morgadinha dos Cannaviaes", "Os fidalgos da Casa Mourisca", "Uma fami%CC%81lia ingleza"),
        "shakespeare":   ("Hamlet", )
    }
}

#import pandas as pd
#import numpy as np
#from collections import Counter
#def count_words_fast(text):
#    text = text.lower()
#    skips = [".", ",", ";", ":", "'", '"']
#    for ch in skips:
#        text = text.replace(ch, "")
#    word_counts = Counter(text.split(" "))
#    return word_counts
#
#def read_book(title_path):
#    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
#    text = text.to_string(index = False)
#    return text
#def word_stats(word_counts):
#    num_unique = len(word_counts)
#    counts = word_counts.values()
#    return (num_unique, counts)
#def word_stats(word_counts):
#    num_unique = len(word_counts)
#    counts = word_counts.values()
#    return (num_unique, counts)
#def word_count_distribution(text):
#    word_counts = count_words_fast(text)
#    count_distribution = Counter(word_counts.values())
#    return count_distribution
#def more_frequent(distribution):
#    counts = sorted(distribution.keys())
#    sorted_frequencies = sorted(distribution.values(), reverse = True)
#    cumulative_frequencies = np.cumsum(sorted_frequencies)
#    more_frequent = 1 - cumulative_frequencies / cumulative_frequencies[-1]
#    return dict(zip(counts, more_frequent))
#hamlets = pd.DataFrame(columns = ("language", "distribution"))
#book_dir = "Books"
#title_num = 1
#for language in book_titles:
#    for author in book_titles[language]:
#        for title in book_titles[language][author]:
#            if title == "Hamlet":
#                inputfile = data_filepath + "Books/" + language + "/" + author + "/" + title + ".txt"
#                text = read_book(inputfile)
#                frequencies = word_count_distribution(text)
#                hamlets.loc[title_num] = language, frequencies
#                title_num += 1
#from matplotlib import pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#from collections import OrderedDict

```

*** =sample_code
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
xlim    = [0, 2e3]
xlabel  = "Word Frequency"
ylabel  = "Probability of Words Being More Frequent"
plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
# show your plot using `plt.show`!
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
xlim    = [0, 2e3]
xlabel  = "Word Frequency"
ylabel  = "Probability of Words Being More Frequent"
plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
plt.show()
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


