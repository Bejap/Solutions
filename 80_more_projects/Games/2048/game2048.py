import copy
import random
from collections import defaultdict

import numpy as np

from utils import colored_tile, transpose, merge_row, boards_differ

INVALID_MOVE_PENALTY = -25


# noinspection PyPackageRequirements
class Game2048:
    """
    Simple implementation of the game 2048
    """
    BOARD_SIZE = 4

    def __init__(self):
        self.score = 0
        self.board = [[0] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.add_new_tile()
        self.add_new_tile()
        self.move_counter = defaultdict(int)
        self.moves = 0
        self.top_tile = 0

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
        self.top_tile = 0

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
            return None
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
            return None

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
            return None

        self.board = transpose(self.board)
        self.board = self.move_left(show=False, add_tile=add_tile)
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
            return None
        self.board = transpose(self.board)
        self.board = self.move_right(show=False, add_tile=add_tile)
        self.board = transpose(self.board)
        if show:
            self.print_board()
        # print("this is score ", self.score)

        return self.board

    def print_board(self):
        print()
        for row in self.board:
            print("\t".join(colored_tile(x) for x in row))
        # print(f"Move: {self.moves}\n")

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

    def step(self, action):
        self.moves += 1
        """
        :param action: 0=up, 1=down, 2=left, 3=right
        :returns: next_state, reward, done
        """

        if self.is_game_over():
            return self.get_state(), -5, True

        before = copy.deepcopy(self.board)
        old_board_max_score = max(cell for row in before for cell in row)

        action_map = {
            0: (self.move_up, 'up'),
            1: (self.move_down, 'down'),
            2: (self.move_left, 'left'),
            3: (self.move_right, 'right')
        }

        move_fn, name = action_map[action]
        merged_board = move_fn(show=False, add_tile=False)
        self.move_counter[name] += 1
        self.move_counter['Sum'] += 1

        valid = boards_differ(merged_board, before)
        if valid:
            self.board = merged_board
            self.add_new_tile()
        else:
            self.board = merged_board
            self.move_counter['Invalid'] += 1
            if self.move_counter['Invalid'] >= 5:
                return self.get_state(), -150, True
            else:
                return self.get_state(), INVALID_MOVE_PENALTY, False


        new_tile_max_score = max(cell for row in self.board for cell in row)

        reward = (new_tile_max_score - old_board_max_score) + (self.moves / 5)

        return self.get_state(), reward, self.is_game_over()

    def get_state(self):
        return [np.log2(cell) if cell > 0 else 0 for row in self.board for cell in row]

    def print_move_summary(self):
        print("Move count summary:")
        for direction in ['up', 'down', 'left', 'right', 'Invalid', 'Sum']:
            print(f"{direction.title()}: {self.move_counter[direction]}")


    def get_valid_actions(self):
        """
        Returns a list of all actions (0=up, 1=down, 2=left, 3=right)
        that would actually change the board.
        """
        valid_actions = []
        # define your action→move-function mapping here
        action_map = {
            0: self.move_up,
            1: self.move_down,
            2: self.move_left,
            3: self.move_right,
        }

        for action, move_fn in action_map.items():
            # snapshot the board
            board_before = copy.deepcopy(self.board)
            # perform the move *without* spawning a new tile
            new_board = move_fn(show=False, add_tile=False)
            # if the move changed the board, it's valid
            if boards_differ(new_board, board_before):
                valid_actions.append(action)
            # restore original board for next test
            self.board = board_before

        return valid_actions


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
