import random


class something:

    def __init__(self):
        self.board = [0] * 8

        self.add_new_tile()

    def add_new_tile(self):
        empty = [r for r in range(8) if self.board[r] == 0]
        if empty:
            r = random.choice(empty)
            self.board[r] = 1

    def move_left(self):
        has_tile = [r for r in range(8) if self.board[r] != 0]
        # print(has_tile)
        for i, tile in enumerate(has_tile):
            print(self.board)
            print(tile)
            tile_position = tile
            while tile_position != 0:
                self.board[tile_position] = 0
                tile_position -= 1
                self.board[tile_position] = 1
            print(self.board)


    def print_board(self):
        print(self.board)


env = something()
env.move_left()
