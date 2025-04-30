import random

class Game2048:
    def __init__(self):
        self.board = [[0]*4 for i in range(4)]
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = random.choice([2]*9 + [4])

    def move_left(self):
        has_tile = [(r, c) for r in range(4) for c in range(4) if self.board != 0]
        for tile in has_tile:
            while tile[0] != 0:
                r = tile[0]
                self.board[r] -= 1



    def move_right(self):
        pass

    def move_up(self):
        pass

    def move_down(self):
        pass

    def print_board(self):
        for row in self.board:
            print("\t".join(str(x).rjust(4) if x != 0 else ".".rjust(4) for x in row))

    def is_game_over(self):
        # Returner True hvis ingen træk muligt
        return False  # placeholder

