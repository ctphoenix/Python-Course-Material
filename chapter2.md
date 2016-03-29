---
title       : Homework 2
description : Exercises for Homework (Week 2)
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf
  
--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1a

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- For our board, we will use a numpy array with dimension 3 by 3.  Make a function create_board() that creates a board, with values of integers 0.
- 
*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
import random
import numpy as np
import matplotlib.pyplot as plt
import time
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board


```

*** =solution
```{python}
import random
import numpy as np
import matplotlib.pyplot as plt
import time
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
```

*** =sct
```{python}
success_msg("Great work!")
```






--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1b

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Players 1 and 2 will take turns changing values of this array from a 0 to a 1 or 2, indicating the number of the player who places there.  Create a function "place" with the first parameter being the current player (an integer 1 or 2) and the second parameter a tuple of length 2 specifying a desired location to place their marker.  Only allow the current player to place a piece on the board (change the board position to their number) if that position is empty (zero).

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
```

*** =solution
```{python}
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
```

*** =sct
```{python}
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1c

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Create a function possibilities(board) that returns a list of all positions on the board that are not occupied (0).

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def possibilities(board):
    return list(zip(*np.where(board == 0)))
```

*** =solution
```{python}
def possibilities(board):
    return list(zip(*np.where(board == 0)))
```

*** =sct
```{python}
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1d

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Create a function random_place(player) that places a marker for the current player at random among all the available positions (those currently set to 0).

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def random_place(board, player):
    selections = possibilities(board)
    if len(selections) > 0:
        selection = random.choice(selections)
        board = place(board, player, selection)
    return board
```

*** =solution
```{python}
def random_place(board, player):
    selections = possibilities(board)
    if len(selections) > 0:
        selection = random.choice(selections)
        board = place(board, player, selection)
    return board
```

*** =sct
```{python}
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1e

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Use the three functions you've made to place three pieces each for players 1 and 2. Print your result.
*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
board = create_board()
for i in range(3):
    for player in [1, 2]:
        board = random_place(board, player)
print(board)
```

*** =solution
```{python}
board = create_board()
for i in range(3):
    for player in [1, 2]:
        board = random_place(board, player)
print(board)
```

*** =sct
```{python}
success_msg("Great work!")
```


--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1f

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Now that players may place their pieces, how will they know they've won?  Make a function row_win that takes the player (integer), and determines if any row consists of only their marker.  Have it return True of this condition is met, and False otherwise.


*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def row_win(board, player):
    winner = False
    if np.any(np.all(board==player,axis=1)):
        return True
    else:
        return False
```

*** =solution
```{python}
def row_win(board, player):
    winner = False
    if np.any(np.all(board==player,axis=1)):
        return True
    else:
        return False
```

*** =sct
```{python}
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1g

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Create a similar function col_win that takes the player (integer), and determines if any column consists of only their marker.  Have it return `True` if this condition is met, and `False` otherwise.


*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def col_win(board, player):
    if np.any(np.all(board==player,axis=0)):
        return True
    else:
        return False
```

*** =solution
```{python}
def col_win(board, player):
    if np.any(np.all(board==player,axis=0)):
        return True
    else:
        return False
```

*** =sct
```{python}
success_msg("Great work!")
```




--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1h

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Finally, create a function `diag_win(player)` that tests if either diagonal of the board consists of only their marker. Have it return `True` if this condition is met, and `False` otherwise.


*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        return True
    else:
        return False
```

*** =solution
```{python}
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        return True
    else:
        return False
```

*** =sct
```{python}
success_msg("Great work!")
```





--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1i

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Create a function `evaluate()` that uses each of these evaluation functions for both players.  If one of them has won, return that player's number.  If the board is full but no one has won, return `-1`.  Otherwise, return `0`.


*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
    if np.all(board != 0):
        winner = -1
    return winner
```

*** =solution
```{python}
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
    if np.all(board != 0):
        winner = -1
    return winner
```

*** =sct
```{python}
success_msg("Great work!")
```

--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1j

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Create a function `play_game()` that creates a board, and alternates `random_place()` for both players, evaluating the board for a winner after every placement. Play the game until one player wins, or the game is a draw.

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def play_game():
    board, winner = create_board(), 0
    while winner == 0:
        for player in [1, 2]:
            board = random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner
```

*** =solution
```{python}
def play_game():
    board, winner = create_board(), 0
    while winner == 0:
        for player in [1, 2]:
            board = random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner
```

*** =sct
```{python}
success_msg("Great work!")
```





--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1k

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Use the `play_game()` function to play 10,000 random games, where Player 1 always goes first.  Use the time library to evaluate how long this takes per game. Plot a histogram of the results.  Does Player 1 win more than Player 2? Does either player win more than each player draws?
*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
start = time.time()
games = [play_game() for i in range(10000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()
```

*** =solution
```{python}
start = time.time()
games = [play_game() for i in range(10000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()
```

*** =sct
```{python}
success_msg("Great work!")
```






--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1k

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- This result is expected --- when guessing at random, it's better to go first.  Let's see if Player 1 can improve their strategy.  Create a function `play_strategic_game()`, where Player 1 always starts with the middle square, and otherwise both players place their markers randomly.

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    while winner == 0:
        for player in [2,1]:
            board = random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner
```

*** =solution
```{python}
def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    while winner == 0:
        for player in [2,1]:
            board = random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner
```

*** =sct
```{python}
success_msg("Great work!")
```



--- type:NormalExercise lang:python xp:100 skills:1
## Exercise 1l

This week, we will create a tic-tac-toe (noughts and crosses) simulator and evaluate basic winning strategies.

*** =instructions
- Use the time libary to evaluate how long each game takes, and compare with the previous game player.  Did Player 1's performance improve?  Does either player win more than each player draws?

*** =hint
-
*** =pre_exercise_code
```{python}
```

*** =sample_code
```{python}
# write your code here!
start = time.time()
games = [play_strategic_game() for i in range(10000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()
```

*** =solution
```{python}
start = time.time()
games = [play_strategic_game() for i in range(10000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()
# Yes, starting in the middle square is a large advantage when play is otherwise
# random.  Also, each game takes less time to play, because each victory is
# decided earlier.  Player 1 wins more than both players draw, and Player 2 wins
# less than both of these outcomes.

```

*** =sct
```{python}
success_msg("Great work!")
```



