# 110911542 Homework3
> this is the code for solving 8 queen problem
```
def is_valid(board, row, col):
    # Check if placing a queen at the given position is valid
    # in terms of row, column, and diagonal conflicts
    for i in range(row):
        # Check column conflicts
        if board[i] == col:
            return False
        # Check diagonal conflicts
        if abs(board[i] - col) == abs(i - row):
            return False
    return True

def solve_eight_queens(board, row=0):
    n = len(board)
    if row == n:
        # All queens have been successfully placed
        return True

    for col in range(n):
        if is_valid(board, row, col):
            # Place the queen at the current position
            board[row] = col

            # Recursively solve for the next row
            if solve_eight_queens(board, row + 1):
                return True

            # Backtrack: Remove the queen from the current position
            board[row] = -1

    return False

def print_board(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

n = 8  # Board size
board = [-1] * n  # Initialize the board

if solve_eight_queens(board):
    print("Solution found:")
    print_board(board)
else:
    print("No solution exists.")
```
> CHATGPT(MODIFY)
