from tkinter import *
import random
import numpy as np
import immutarray as ia
from queue import PriorityQueue
import time

# An immutable array is an array that has been made immutable, and thus has a hash value and can work as a key in a dictionary (I do this)

# Solution board
solution = ia.ImmutArray([
    [1,2,3],
    [8,0,4],
    [7,6,5]
    ])

# A move is a node with a parent move, the tile moved (numbered 1-8), the heurstic for its board state, and the board state
# It is comparable to other moves, where it is LESS than another move if it's heuristic is GREATER
class Move:
    g=None
    tile_moved=None
    heuristic=None
    prior_move=None
    board = None

    # Make it comparable based on f-value for priority queue
    def __lt__(self, other):
        return (self.g + self.heuristic) < (other.g + other.heuristic)
    
    def __eq__(self, other):
        return (self.g + self.heuristic) == (other.g + other.heuristic)
    
# Moves to explore is list of moves. It will choose the move with the lowest f value cost
moves_to_explore = PriorityQueue()

# {board, cost}
# Store all of the discovered board states and the cost to get there, only keep one of each board state with the lowest cost to getting there.
# In order to use board_state as a key, I had to use an immutable array, referenced in the other python file.
state_graph = {}

# Replacing the numpy where function because I am using immutable arrays and np.where won't work on them. Plus, my gameboard is always 3x3 so my where function
# is much simpler, especially since all the values on the board are unique. Complexity doesn't win any prizes
def where(arr, value):
    for x in range (0,3):
        for y in range(0, 3):
            if arr[x, y] == value:
                return (x, y)

# Randomly initialize the board and create the first move, which has no tile moved
def init():
    list = [1,2,3,4,5,6,7,8,0]
    random.shuffle(list)
    i = 0
    initial_board = np.zeros((3, 3))
    for x in range(0,3):
        for y in range(0,3):
            initial_board[x][y] = list[i]
            i = i+1

    first_move = Move()
    first_move.board = ia.ImmutArray(initial_board)
    first_move.heuristic = compute_total_distance(first_move.board)
    first_move.g = 0
    state_graph.update({first_move.board: first_move.g})
    moves_to_explore.put((first_move.heuristic, first_move))
    return initial_board
 
# Provided coordinates and the gameboard, calculate the distance of the value at those coordinates from the correct coordinates
def tile_distance_from_solution(arr, x, y):
    value = arr[x][y]
    correct_coords = where(solution, value)
    return abs(correct_coords[0] - x) + abs(correct_coords[1] - y)

# Calculate the total distance from the solution and return the value
def compute_total_distance(arr):
    distance = 0
    for x in range (0, 3):
        for y in range (0, 3):
            distance = distance + tile_distance_from_solution(arr, x, y)
    return distance

# Given a prior move, find tiles adjacent to 0 given the prior move's board state and create moves from that state
def expand_move(prior_move):
    zero_coords = where(prior_move.board, 0)
    for x in range(-1,2):
        for y in range(-1, 2):
            if not (abs(x) == abs(y)): # Never move the diagonals
                moved_index = [zero_coords[0] + x, zero_coords[1] + y]
                if moved_index[0] >= 0 and moved_index[0] <= 2 and moved_index[1] >= 0 and moved_index[1] <= 2: # Don't move something out of bounds
                    tile_moved = prior_move.board[moved_index[0]][moved_index[1]]
                    move = make_move(prior_move, tile_moved)
                    if move.board not in state_graph or (move.g < state_graph.get(move.board)):
                        state_graph.update({move.board: move.g})
                        moves_to_explore.put((move.heuristic, move))



# Make a move given the prior move and the tile being moved for the new move
def make_move(prior_move, tile_moved):
    move = Move()
    move.tile_moved = tile_moved
    move.prior_move = prior_move
    zero_coords = where(prior_move.board, 0)
    move_coord = where(prior_move.board, move.tile_moved)
    move.board = np.copy(prior_move.board)
    move.board[zero_coords[0],zero_coords[1]] = move.tile_moved
    move.board[move_coord[0], move_coord[1]] = 0
    move.board = ia.ImmutArray(move.board)
    move.heuristic = compute_total_distance(move.board)
    move.g = prior_move.g + 1
    return move

# Do one iteration of the algorithm and return the last move if we have solved it
def iteration():
    # Find the best move to make
    item = moves_to_explore.get()
    print("Heuristic of best move: ",item[0])
    best_move = item[1]

    # If the heuristic of the best move is 0, we have solved the puzzle
    if best_move.heuristic == 0:
        print("Board has been solved")
        return best_move
    else: # Otherwise we must continue to suffer
        expand_move(best_move)
        return None


# Callback function, solves the puzzle and shows the moves to get to the solution
def solve(rows, distance_display):
    last_move = None
    while last_move is None:
        last_move = iteration()

    solution_ordered = []
    while last_move.prior_move is not None:
        solution_ordered.append(last_move)
        last_move = last_move.prior_move
    solution_ordered.reverse()

    for last_move in solution_ordered:
        time.sleep(0.5)
        for x in range(0,3):
            for y in range(0,3):
                rows[x][y].configure(text=(str(int(last_move.board[x][y]))), bg='white')
                if last_move.board[x][y] == 0:
                    rows[x][y].configure(bg='red')
                rows[x][y].update()
        distance_display.configure(text=(("Total distance: " + str(last_move.heuristic))))

initial_board = init()

ws = Tk()
ws.title('A* Algorithm')
ws.geometry('500x500')
ws.config(bg='#9FD996')

rows = []
for x in range(0,3):
    cols = []
    for y in range(0,3):
        e = Label(ws,text=(str(int(initial_board[x][y]))), font='Helvetica 14 bold', width=10,height=6, relief="groove")
        e.grid(row=x, column=y, sticky=NSEW,)
        cols.append(e)
    rows.append(cols)
distance_display = Label(ws,text=(("Total distance: " + str(int(compute_total_distance(initial_board))))),width=20,height=4, relief="groove")
distance_display.grid(row=3,column=0,sticky=NSEW,)
btn= Button(ws, text= "Solve", command= lambda:solve(rows, distance_display))
btn.grid(row=4, column=0, sticky=NSEW,)
ws.mainloop()
