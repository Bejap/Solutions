import random
import copy
from utils import colored_tile, transpose, merge_row, boards_differ
from collections import defaultdict
import numpy as np


INVALID_MOVE_PENALTY = -2
class Game2048:
    """
    Simple implementation of the game 2048
    """
    BOARD_SIZE = 3

    def __init__(self):
        self.score = 0
        self.board = [[0] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.add_new_tile()
        self.add_new_tile()
        self.move_counter = defaultdict(int)
        self.moves = 0

    def add_new_tile(self):
        """
        Adds a new tile to the board of the value 2 or 4
        """
        empty = [(r, c) for r in range(self.BOARD_SIZE) for c in range(self.BOARD_SIZE) if self.board[r][c] == 0]
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = random.choice([2] * 9 + [4])

    def move_left(self, show=True, add_tile=True):
        """
        Moves all values to the left, if valid
        """
        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)

        for i in range(self.BOARD_SIZE):
            row = self.board[i]
            self.board[i] = merge_row(row)

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
        if show:
            self.print_board()
            # print("this is score ", self.score)

        return self.board

    def move_right(self, show=True, add_tile=True):
        """
        Moves all values to the right
        """

        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)
        for i in range(self.BOARD_SIZE):
            row = self.board[i][::-1]  # reverse row
            merged = merge_row(row)
            self.board[i] = merged[::-1]  # reverse back

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
        if show:
            self.print_board()
            # print("this is score ", self.score)

        return self.board

    def move_up(self, show=True, add_tile=True):
        """
        Moves all values to the up
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        self.move_left(show=False, add_tile=add_tile)
        self.board = transpose(self.board)
        if show:
            self.print_board()
        # print("this is score ", self.score)

        return self.board

    def move_down(self, show=True, add_tile=True):
        """
        Moves all values to the down
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        self.move_right(show=False, add_tile=add_tile)
        self.board = transpose(self.board)
        if show:
            self.print_board()
        # print("this is score ", self.score)

        return self.board

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
            for i in range(self.BOARD_SIZE - 1):
                if row[i] == row[i + 1]:
                    return False

        for c in range(self.BOARD_SIZE):
            for r in range(self.BOARD_SIZE - 1):
                if self.board[r][c] == self.board[r + 1][c]:
                    return False

        return True  # placeholder

    def reset(self):
        """
        resets the board
        """
        self.board = [[0] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.add_new_tile()
        self.add_new_tile()
        self.moves = 0
        self.print_board()
        self.move_counter = defaultdict(int)
        self.score = 0

    def step(self, action):
        """
        :param action: 0=up, 1=dwon, 2=left, 3=right
        :returns: next_state, reward, done
        """

        if self.is_game_over():
            return self.get_state(), -100, True

        old_score = 0
        before = copy.deepcopy(self.board)
        for i in range(self.BOARD_SIZE):
            old_score += sum(before[i])

        action_map = {
            0: (self.move_up, 'up'),
            1: (self.move_down, 'down'),
            2: (self.move_left, 'left'),
            3: (self.move_right, 'right')
        }

        move_fn, name = action_map[action]
        new_board = move_fn(show=False, add_tile=False)
        self.move_counter[name] += 1
        new_score = 0

        for i in range(self.BOARD_SIZE):
            new_score = sum(self.board[i])
        reward = new_score - old_score

        self.score += reward
        if boards_differ(new_board, before):
            self.board = new_board
            self.add_new_tile()
        else:
            self.board = new_board  # stays the same
            reward = INVALID_MOVE_PENALTY

        return self.get_state(), reward, self.is_game_over()

    def get_state(self):
        return [np.log2(cell) if cell > 0 else 0 for row in self.board for cell in row]

    def print_move_summary(self):
        print("Move count summary:")
        for direction in ['up', 'down', 'left', 'right', 'Invalid']:
            print(f"{direction.title()}: {self.move_counter[direction]}")


if __name__ == '__main__':
    game = Game2048()
    game.print_board()
    while not game.is_game_over():
        print("UP")
        state, rew, done = game.step(0)
        print(game.print_board())
        print('state ', state, 'reward ', rew, 'done', done)
        print("\nLEFT")
        state, rew, done = game.step(2)
        print(game.print_board())
        print('\nstate ', state, 'reward ', rew, 'done', done)
        print("\nRIGHT")
        state, rew, done = game.step(3)
        print(game.print_board())
        print('\nstate ', state, 'reward ', rew, 'done', done)
        print("\nDOWN")
        state, rew, done = game.step(1)
        print(game.print_board())
        print('\nstate ', state, 'reward ', rew, 'done', done)

    print("Game over")
    print(game.print_move_summary())
