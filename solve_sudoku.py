#### SUDOKU SOLVING ALGORITHM
# We know that sudoku is a 9x9 2D matrix

def foundInCol(arr, col, num):  # Checks if num is aldready present in the column
    for i in range(9):
      if(arr[i][col] == num):
        return True
    return False


def foundInRow(arr, row, num):  # Checks if num is aldready present in the row
    for i in range(9):
      if(arr[row][i] == num):
        return True
    return False


def foundInBox(arr, row, col, num): # Checks if the num exists in the 3x3 grid
    startRow = row - (row % 3)
    startCol = col - (col % 3)
    for i in range(3):
      for j in range(3):
        if(arr[i + startRow][j + startCol] == num):
          return True
    return False

def isSafe(arr, row, col, num):
    return ((not foundInRow(arr, row, num)) and (not foundInCol(arr, col, num)) and (not foundInBox(arr, row, col, num)))

def foundEmptyCell(arr, loc): # Finds the location of the next empty cell 
    for i in range(9):
      for j in range(9):
        if(arr[i][j] == 0):
          loc[0] = i  # loc[0] will give the empty cell row
          loc[1] = j  # loc[1] will give the empty cell column
          return True
    return False

def solveSudoku(arr):

    l = [0,0]

    if(not foundEmptyCell(arr, l)): # Returns True when all spaces are filled by us
        return True

    row = l[0]  # Assigns the empty location
    col = l[1]  # got from the above function

    for num in range(1, 10):
        if(isSafe(arr, row, col, num)):
            arr[row][col] = num
            if(solveSudoku(arr)):
                return True
            arr[row][col] = 0       # If a num is safe, but there doesn't exist a solution with it; the location must be set to 0 for further iterations of num 

    return False  # Backtracking