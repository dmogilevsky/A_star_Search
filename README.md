# Solving the 8 Tile Puzzle Using A Star Search
![image](https://user-images.githubusercontent.com/70275882/163489034-8e118340-d663-45b8-b941-b08a808e191e.png)
## Introduction
The 8 tile puzzle is a well known sliding puzzle. It contains 8 numbered tiles numbered 1-8, and an empty tile. In order to solve the puzzle,
the tiles must be moved from a given starting position to a specified ending position, however, tiles can not "swap" places with each other. They may only move into the empty space.

## Pre-requisites
Python 3 must be installed along with the Tkinter, Numpy, and matplotlib libraries

## Running the Puzzle Solver
1. Clone or copy the astar.py file onto your local machine.
2. Run python3 astar.py in the directory you have placed the file

![image](https://user-images.githubusercontent.com/70275882/163489435-5fca9b49-c9f9-44f6-a719-9eb807f7bc82.png)
## Using the Solver
When the solver is started, a random game board will be initialized alongside an empty plot. The plot will graph the search space by plotting each
board state onto the plot according to it's heuristic (calculated distance from solved state) and it's distance from the starting position (calculated as the number of moves to get to this state). Additionally, the graph will show lines representing which board states each board state came from.

You can configure the solver in the following ways:
- Check Live Board/Graph Updates in order to see the board states/graph update as the solver searches through the solution space (Please note, this can significantly slow the speed of the solver)
- Change the delay time (measured in seconds) which will change the delay between each move shown in the solution once the solution is found

When a solution is found, the solver will go back to the board's starting position and move through the solution. The lines on the graph will be updated, showing the path of the solution.
