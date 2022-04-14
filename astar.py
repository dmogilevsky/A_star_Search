from tkinter import *
import numpy as np
from queue import PriorityQueue
import time
import matplotlib.pyplot as plt

# Solution board
solution = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]
    ])

# Moves to explore is list of moves. It will choose the move with the lowest f value cost
moves_to_explore = PriorityQueue()

# {board, move}
# Store all of the discovered board states and the cost to get there, only keep one of each board state with the lowest cost to getting there.
state_graph = {}

# Need this to be a global
fig = plt.figure(figsize=(10,5))
plot, = plt.plot([], [])
pointset = set()
lineset = set()
delayEntry = None
live_board=False # Configuring whether to show the board fly through all of the checked states during solving
live_graph=False # Configuring whether to plot the graph sequentially or plot the search space all at once at the end
delay_time=0.5   # Configure the delay between each move shown for the solution

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
        return (self.f()) > (other.f())

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

# https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
# ########################################################################
# A utility function to count
# inversions in given array 'arr[]'
def getInvCount(arr):
    inv_count = 0
    empty_value = 0
    for i in range(9):
        for j in range(i + 1, 9):
            if arr[j] != empty_value and arr[i] != empty_value and arr[i] > arr[j]:
                inv_count += 1
    return inv_count

# This function returns true
# if given 8 puzzle is solvable.
def isSolvable(puzzle) :
    # Count inversions in given 8 puzzle
    inv_count = getInvCount([j for sub in puzzle for j in sub])
    # return true if inversion count is even.
    return (inv_count % 2 == 0)
###########################################################################

# Randomly initialize the board and create the first move, which has no tile moved
def init():
    # Initialize plot
    ax = fig.gca()
    ax.set_ylim(0,28)
    ax.set_xlim(0,30)
    plt.xlabel('G: Distance From Start')
    plt.ylabel('H: Heuristic')
    plt.title('Graph of Search Space')

    # Create board and first move
    solvable = False
    initial_board = None
    while not solvable:
        initial_board = np.arange(9)
        np.random.shuffle(initial_board)
        initial_board = initial_board.reshape((3,3))
        solvable = isSolvable(initial_board)

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
    for x in range (3):
        for y in range (3):
            #if arr[x][y] != solution[x][y]: distance = distance + 1
            distance = distance + tile_distance_from_solution(arr, x, y)
    return distance[0]

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
                        liveUpdate(move)
                        state_graph.update({boardKey: move})   # Then put the new move in the state graph
                        moves_to_explore.put((move.f(), move)) # And in the moves to explore list



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
    best_move = best_move = moves_to_explore.get()[1]
    if best_move.g > state_graph.get(boardToKey(best_move.board)).g:
        print("ignored")
        return None # If we've already gotten to this with a lower cost before, don't bother with this move
    print("F:",best_move.f(),"H:",best_move.h,"G:",best_move.g,"Moves to Explore:",moves_to_explore.qsize(),"# of States:",len(state_graph))
    if best_move.h == 0:    # If the heuristic of the best move is 0, we have solved the puzzle
        return best_move
    else:                   # Otherwise we must continue to suffer
        expand_move(best_move)
        return None


# Callback function, solves the puzzle and shows the moves to get to the solution
def solve():
    last_move = None
    while last_move is None:
        last_move = iteration()

    print("Solved")

    solution_ordered = []
    while last_move.prior_move is not None:
        solution_ordered.insert(0, last_move)
        last_move = last_move.prior_move

    print("Backtraced solution")

    if not live_graph: addAllStateGraphToPlot()
    plt.show(block=False)

    for last_move in solution_ordered:
        time.sleep(float(delayEntry.get()))
        addMoveToPlotAndUpdate(last_move, True)
        updateBoard(last_move)


def addAllStateGraphToPlot():
    for move in state_graph.values():
        addMoveToPlot(move, False)

# Add a given move to the plot by plotting both the point representing the move's g and h values, as well as a line
# from the parent move's point
def addMoveToPlot(move, backtrace):
    color = 'ro'
    if backtrace: color = 'black'
    if (move.g, move.h) not in pointset or backtrace:
        pointset.add((move.g, move.h))
        plt.plot(move.g, move.h, color)
    if move.prior_move is not None:
        line = ((move.prior_move.g, move.g), (move.prior_move.h, move.h))
        if (line not in lineset or backtrace) and move.prior_move is not None:
            lineset.add(line)
            if color == 'ro': plt.plot([move.prior_move.g, move.g], [move.prior_move.h, move.h])
            else: plt.plot([move.prior_move.g, move.g], [move.prior_move.h, move.h], color, linewidth=4)

# Add a move to the plot and update it in real time
def addMoveToPlotAndUpdate(move, backtrace):
    addMoveToPlot(move, backtrace)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# Update the board to match the given move
def updateBoard(move):
        for x in range(3):
            for y in range(3):
                if move.board[x][y] == 0:
                    rows[x][y].configure(bg='red', text='')
                else:
                    rows[x][y].configure(text=(str(int(move.board[x][y]))), bg='white')
                rows[x][y].update()
        distance_display.configure(text=(("Total distance: " + str(move.h))))

# Do a live update of the graph and board based on whether or not lvie update's were enabled
def liveUpdate(move):
    if live_graph:
        addMoveToPlotAndUpdate(move, False)
    if live_board:
        updateBoard(move)

initial_board = init()

ws = Tk()
ws.title('A* Algorithm')
ws.geometry('350x450')
ws.config(bg='#9FD996')

rows = []
for x in range(0,3):
    cols = []
    for y in range(0,3):
        e = Label(ws,text=(str(int(initial_board[x][y]))), font='Helvetica 14 bold', width=10,height=6, relief="groove")
        if initial_board[x][y] == 0:
            e.configure(text='')
        e.grid(row=x, column=y, sticky=NSEW,)
        cols.append(e)
    rows.append(cols)
buttonFrame = Frame(ws)
buttonFrame.grid(row=3,column=0,columnspan=3)
distance_display = Label(buttonFrame,text=(("Total distance: " + str(int(compute_total_distance(initial_board))))),width=15,height=4, relief="groove")
distance_display.pack(side=LEFT)
Button(buttonFrame, text= "Solve", command= lambda:solve()).pack(side=LEFT,fill='y')

def set_live_board():
    global live_board
    if live_board:
        live_board = False
    else:
        live_board = True

def set_live_graph():
    global live_graph
    if live_graph:
        live_graph = False
    else:
        live_graph = True

Checkbutton(buttonFrame, text='Live Board Updates', command=set_live_board).pack()
Checkbutton(buttonFrame, text='Live Graph Updates', command=set_live_graph).pack()

# Delay Time Entry box
entryFrame = Frame(buttonFrame)
entryFrame.pack()
Label(entryFrame, text='Delay Time').pack(side=LEFT)
delayEntry = Entry(entryFrame, width=7)
delayEntry.insert(0, delay_time)
delayEntry.pack(side=RIGHT)

plot, = plt.plot(0, compute_total_distance(initial_board), 'ro')
plt.show(block=False)
ws.mainloop()
