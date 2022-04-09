from tkinter import *
import numpy as np
from queue import PriorityQueue
import time
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTk, NavigationToolbar2Tk)
from IPython.display import clear_output

# Solution board
solution = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]
    ])

ws = Tk()

# Convert a board to a key, allowing for use as a key in a dictionary
def boardToKey(myboard):
    return(''.join([str(x) for x in myboard.flatten().tolist()]))

def prettyPrintBoard(key):
    return(
    "["+key[0]+","+key[1]+","+key[2]+"]\n"
    "["+key[3]+","+key[4]+","+key[5]+"]\n"
    "["+key[6]+","+key[7]+","+key[8]+"]\n"
    )

# Convert a string to a board
def keyToBoard(myKey):
    board = np.array(list(myKey))
    board.reshape((3, 3))
    return board

# A move is a node with a parent move, the tile moved (numbered 1-8), the heurstic for its board state, and the board state
# It is comparable to other moves, where it is LESS than another move if it's heuristic is GREATER
class Move:
    g=None
    h=None
    tile_moved=None
    prior_move=None
    board = None

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return (self.f()) < (other.f())
    
    def __eq__(self, other):
        return (self.f()) == (other.f())
    
# Moves to explore is list of moves. It will choose the move with the lowest f value cost
moves_to_explore = PriorityQueue()

# {board, move}
# Store all of the discovered board states and the cost to get there, only keep one of each board state with the lowest cost to getting there.
state_graph = {}

# Randomly initialize the board and create the first move, which has no tile moved
def init():
    initial_board = np.arange(9)
    np.random.shuffle(initial_board)
    initial_board = initial_board.reshape((3,3))

    first_move = Move()
    first_move.board = initial_board
    first_move.h = compute_total_distance(first_move.board)
    first_move.g = 0
    state_graph.update({boardToKey(first_move.board): first_move})
    moves_to_explore.put((first_move.f(), first_move))
    return initial_board
 
# Provided coordinates and the gameboard, calculate the distance of the value at those coordinates from the correct coordinates
def tile_distance_from_solution(arr, x, y):
    value = arr[x][y]
    correct_coords = np.where(solution==value)
    return abs(correct_coords[0] - x) + abs(correct_coords[1] - y)

# Calculate the total distance from the solution and return the value
def compute_total_distance(arr):
    distance = 0
    for x in range (0, 3):
        for y in range (0, 3):
            if arr[x][y] != solution[x][y]: distance = distance + 1
            #distance = distance + tile_distance_from_solution(arr, x, y)
    return distance

# Given a prior move, find tiles adjacent to 0 given the prior move's board state and create moves from that state
def expand_move(prior_move):
    zero_coords = np.where(prior_move.board==0)
    for x in range(-1,2):
        for y in range(-1, 2):
            if (abs(x) != abs(y)): # Never move the diagonals
                moved_index = (zero_coords[0] + x, zero_coords[1] + y)
                if moved_index[0] >= 0 and moved_index[0] <= 2 and moved_index[1] >= 0 and moved_index[1] <= 2: # Don't move something out of bounds
                    tile_moved = prior_move.board[moved_index]
                    move = make_move(prior_move, tile_moved)
                    boardKey = boardToKey(move.board)
                    if boardKey not in state_graph or (move.g < state_graph.get(boardKey).g):
                        state_graph.update({boardKey: move})
                        moves_to_explore.put((move.f(), move))



# Make a move given the prior move and the tile being moved for the new move
def make_move(prior_move, tile_moved):
    move = Move()
    move.tile_moved = tile_moved
    move.prior_move = prior_move
    move.board = np.copy(prior_move.board)
    move.board[np.where(prior_move.board==0)] = move.tile_moved
    move.board[np.where(prior_move.board==tile_moved)] = 0
    move.h = compute_total_distance(move.board)
    move.g = prior_move.g + 1
    return move

# Do one iteration of the algorithm and return the last move if we have solved it
def iteration():
    

    best_move = moves_to_explore.get()[1]

    print("Lowest Cost:",best_move.f(),"Moves to Explore:",moves_to_explore.qsize(),"# of States:",len(state_graph))
    # If the heuristic of the best move is 0, we have solved the puzzle
    if best_move.h == 0:
        #print("Board has been solved")
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

    addAllStateGraphToPlot()
    plt.show(block=False)

    for last_move in solution_ordered:
        time.sleep(0.5)
        for x in range(0,3):
            for y in range(0,3):
                rows[x][y].configure(text=(str(int(last_move.board[x][y]))), bg='white')
                if last_move.board[x][y] == 0:
                    rows[x][y].configure(bg='red')
                rows[x][y].update()
        distance_display.configure(text=(("Total distance: " + str(last_move.h))))

def addAllStateGraphToPlot():
    for move in state_graph.values():
        addMoveToPlot(move)

def addMoveToPlot(move):
    plt.plot(move.g, move.h, 'ro')
    if move.prior_move is not None:
        plt.plot([move.prior_move.g, move.g], [move.prior_move.h, move.h])


initial_board = init()

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
buttonFrame = Frame(ws)
buttonFrame.grid(row=3,column=0,columnspan=3)
distance_display = Label(buttonFrame,text=(("Total distance: " + str(int(compute_total_distance(initial_board))))),height=4, relief="groove")
distance_display.pack(side=LEFT)
btn= Button(buttonFrame, text= "Solve", command= lambda:solve(rows, distance_display))
btn.pack(side=LEFT)

#plt.plot(0, compute_total_distance(initial_board),'ro')
#plt.plot([3, 5], [1, 6],color="green")
plt.ylim([0, 10])
plt.xlabel('G: Distance From Start')
plt.ylabel('H: Heuristic')
#plt.plot([0, 1], [compute_total_distance(initial_board), 7])
#plt.show()

# plotFrame = Frame(ws, width=500, height=500)
# plotFrame.grid(row=0,column=3,rowspan=5)
# fig = Figure(figsize = (5, 5),dpi = 100)
# ax = fig.add_subplot(111)
# t = np.arange(0.0,3.0,0.01)
# s = np.sin(np.pi*t)
# ax.plot(t,s)
# canvas = FigureCanvasTk(fig, plotFrame)  
# canvas.draw()
# canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
# toolbar = NavigationToolbar2Tk(canvas, plotFrame)
# toolbar.pack()
# toolbar.update()
# canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
# canvas.draw_idle()
ws.mainloop()
