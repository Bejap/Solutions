import random

class Game2048:
    def __init__(self):
        self.board = [[0]*4 for i in range(4)]
        self.add_new_tile()
        self.add_new_tile()
        self.moves = 0

    def add_new_tile(self):
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = random.choice([2]*9 + [4])



    def move_left(self, show=True):
        if not self.is_game_over():
            for i in range(4):
                row = self.board[i]
                self.board[i] = self._merge_row(row)
            self.add_new_tile()
            if show:
                self.print_board()


    def move_right(self, show=True):
        if not self.is_game_over():
            for i in range(4):
                row = self.board[i][::-1]  # reverse row
                merged = self._merge_row(row)
                self.board[i] = merged[::-1]  # reverse back
            self.add_new_tile()
            if show:
                self.print_board()

    def move_up(self):
        if not self.is_game_over():
            self.board = self._transpose(self.board)
            self.move_left(show=False)
            self.board = self._transpose(self.board)
            self.print_board()

    def move_down(self):
        if not self.is_game_over():
            self.board = self._transpose(self.board)
            self.move_right(show=False)
            self.board = self._transpose(self.board)
            self.print_board()

    def _merge_row(self, row):
        new_row = [x for x in row if x != 0]

        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                new_row[i + 1] = 0
                i += 2
            else:
                i += 1

        # Step 3: Remove zeros again and pad with 0s to the right
        final = [x for x in new_row if x != 0]
        final += [0] * (4 - len(final))
        return final


    def print_board(self):
        self.moves += 1
        for row in self.board:
            print("\t".join(str(x).rjust(4) if x != 0 else ".".rjust(4) for x in row))
        print(self.moves, "\n")

    def is_game_over(self):
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

    def _transpose(self, board):
        k = [row.copy() for row in board]
        for i in range(4):
            for j in range(4):
                k[j][i] = board[i][j]

        return k



if __name__ == '__main__':
    game = Game2048()
    game.print_board()
    while not game.is_game_over():
        print("UP")
        game.move_up()
        print("LEFT")
        game.move_left()
        print("RIGHT")
        game.move_right()
        print("DOWN")
        game.move_down()

