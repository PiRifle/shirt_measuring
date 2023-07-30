def is_outside(row: int, col: int, r: int, c: int):
    if (r >= row or c >= col or r < 0 or c < 0):
        return True
    return False
 
# Function to rotate in clockwise manner
def next_turn(previous_direction: str, r: int, c: int):
    if (previous_direction == 'u'):
        # turn_right
        c += 1
        previous_direction = 'r'
    elif (previous_direction == 'r'):
        # turn_down
        r += 1
        previous_direction = 'd'
    elif (previous_direction == 'd'):
        # turn_left
        c -= 1
        previous_direction = 'l'
    elif (previous_direction == 'l'):
        # turn_up
        r -= 1
        previous_direction = 'u'
    return previous_direction, r, c
 
# Function to move in the same direction
# as its prev_direction
def move_in_same_direction(previous_direction: str, r: int, c: int):
    if (previous_direction == 'r'):
        c += 1
    elif (previous_direction == 'u'):
        r -= 1
    elif(previous_direction == 'd'):
        r += 1
    elif(previous_direction == 'l'):
        c -= 1
    return r, c
 
# Function to find the spiral order of
# of matrix according to given rules
def spiralMatrix(rows:int, cols:int, r:int, c:int):
    # For storing the co-ordinates
    previous_direction = 'r'
 
    # For keeping track of no of steps
    # to go without turn
    turning_elements = 2
 
    # Count is for counting total cells
    # put in the res
    count = 0
 
    # Current_count is for keeping track
    # of how many cells need to
    # traversed in the same direction
    current_count = 0
 
    # For keeping track the number
    # of turns we have made
    turn_count = 0
    limit = rows * cols
 
    while (count < limit):
 
        # If the current cell is within
        # the board
        if (is_outside(rows, cols, r, c) == False):
            yield r, c
            count += 1
 
        current_count += 1
 
        # After visiting turning elements
        # of cells we change our turn
        if (current_count == turning_elements):
 
            # Changing our direction
            # we have to increase the
            # turn count
            turn_count += 1
 
            # In Every 2nd turn increasing
            # the elements the turn visiting
            if (turn_count == 2):
                turning_elements += 1
 
            # After every 3rd turn reset
            # the turn_count to 1
            elif (turn_count == 3):
                turn_count = 1
 
            # Changing direction to next
            # direction based on the
            # previous direction
            store = next_turn(previous_direction, r, c)
            previous_direction = store[0]
            r = store[1]
            c = store[2]
             
            # Reset the current_count
            current_count = 1
 
        else:
            store = move_in_same_direction(
                previous_direction, r, c)
            r = store[0]
            c = store[1]
