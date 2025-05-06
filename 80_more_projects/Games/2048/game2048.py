import random
import copy
from utils import colored_tile, transpose, merge_row, boards_differ

class Game2048:
    """
    Simple implementation of the game 2048
    """

    def __init__(self):
        self.score = 0
        self.board = [[0] * 4 for _ in range(4)]
        self.add_new_tile()
        self.add_new_tile()
        self.moves = 0

    def add_new_tile(self):
        """
        Adds a new tile to the board of the value 2 or 4
        """
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = random.choice([2] * 9 + [4])

    def move_left(self, show=True, add_tile=True):
        """
        Moves all values to the left, if valid
        """
        score = 0
        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)

        for i in range(4):
            row = self.board[i]
            self.board[i], score = merge_row(row)

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
            self.score += score
        if show:
            self.print_board()
            print("this is score ", self.score)

    def move_right(self, show=True, add_tile=True):
        """
        Moves all values to the right
        """
        score = 0
        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)
        for i in range(4):
            row = self.board[i][::-1]  # reverse row
            merged, score = merge_row(row)
            self.board[i] = merged[::-1]  # reverse back

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
            self.score += score
        if show:
            self.print_board()
            print("this is score ", self.score)

    def move_up(self, add):
        """
        Moves all values to the up
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        self.move_left(show=False, add_tile=add)
        self.board = transpose(self.board)
        self.print_board()
        print("this is score ", self.score)

    def move_down(self, add):
        """
        Moves all values to the down
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        self.move_right(show=False, add_tile=add)
        self.board = transpose(self.board)
        self.print_board()
        print("this is score ", self.score)

    def print_board(self):
        self.moves += 1
        print()
        for row in self.board:
            print("\t".join(colored_tile(x) for x in row))
        print(f"Move: {self.moves}\n")

    def is_game_over(self):
        """
        Check if there are no valid moves left (no empty cells and no adjacent matches).
        """
        for row in self.board:
            if 0 in row:
                return False

        for row in self.board:
            for i in range(3):
                if row[i] == row[i + 1]:
                    return False

        for c in range(4):
            for r in range(3):
                if self.board[r][c] == self.board[r + 1][c]:
                    return False

        return True  # placeholder

    def reset(self):
        self.board = [[0] * 4 for _ in range(4)]
        self.add_new_tile()
        self.add_new_tile()
        self.moves = 0
        self.print_board()


if __name__ == '__main__':
    game = Game2048()
    game.print_board()
    while not game.is_game_over():
        print("UP")
        game.move_up(True)
        print("LEFT")
        game.move_left()
        print("RIGHT")
        game.move_right()
        print("DOWN")
        game.move_down(True)
