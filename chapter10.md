---
title       : Case Study 7 Part 2 - Movie Analysis: Modeling
description : [The Movie Database](https://www.kaggle.com/tmdb/tmdb-movie-metadata) is a database of 5000 movies catalogued by [IMDB](http://www.imdb.com/). The information available about each movie is its budget, revenue, rating, actors and actresses, etc. In this Case Study, we will use this dataset to predict whether any information about a movie can predict how much money a movie will make. We will also attempt to predict whether a movie will make more money than is spent on its budget.
--- type:NormalExercise lang:python xp:100 skills:2 key:07ea54b341

To perform prediction and classification, we will primarily use the two models we recently discussed: generalized linear regression, and random forests. We will use linear regression to predict revenue, and logistic regression to classify whether a movie made a profit. Random forests come equipped with both a regression and classification mode, and we will use both of them to predict revenue and whether or not a movie made a profit.

## Exercise 1