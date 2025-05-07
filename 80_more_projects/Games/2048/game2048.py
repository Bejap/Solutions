import random
import copy
from utils import colored_tile, transpose, merge_row, boards_differ


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
        self.score = 0
        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)

        for i in range(self.BOARD_SIZE):
            row = self.board[i]
            self.board[i] = merge_row(row)
            if self.score < max(self.board[i]):
                self.score = max(self.board[i])

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
        if show:
            self.print_board()
            # print("this is score ", self.score)

        return self.board, self.score

    def move_right(self, show=True, add_tile=True):
        """
        Moves all values to the right
        """
        score = 0
        if self.is_game_over():
            return

        before = copy.deepcopy(self.board)
        for i in range(self.BOARD_SIZE):
            row = self.board[i][::-1]  # reverse row
            merged = merge_row(row)
            self.board[i] = merged[::-1]  # reverse back
            if score < max(self.board[i]):
                score = max(self.board[i])

        if boards_differ(self.board, before) and add_tile:
            self.add_new_tile()
        if show:
            self.print_board()
            # print("this is score ", self.score)

        return self.board, score

    def move_up(self, add):
        """
        Moves all values to the up
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        _, reward = self.move_left(show=False, add_tile=add)
        self.board = transpose(self.board)
        self.print_board()
        # print("this is score ", self.score)

        return self.board, reward

    def move_down(self, add):
        """
        Moves all values to the down
        """
        if self.is_game_over():
            return
        self.board = transpose(self.board)
        _, reward = self.move_right(show=False, add_tile=add)
        self.board = transpose(self.board)
        self.print_board()
        # print("this is score ", self.score)

        return self.board, reward

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

    def step(self, action):
        """
        :param action: 0=up, 1=dwon, 2=left, 3=right
        :returns: next_state, reward, done
        """

        if self.is_game_over():
            return self.get_state(), -100, True

        before = copy.deepcopy(self.board)

        if action == 0:
            new_board, b_score = self.move_up(True)
            reward = b_score - self.score
            self.board = new_board
        elif action == 1:
            new_board, b_score = self.move_down(True)
            reward = b_score - self.score
            self.board = new_board
        elif action == 2:
            new_board, b_score = self.move_left()
            reward = b_score - self.score
            self.board = new_board
        elif action == 3:
            new_board, b_score = self.move_right()
            reward = b_score - self.score
            self.board = new_board
        else:
            raise ValueError("Invalid action")

        if not boards_differ(new_board, before):
            reward = -10

        return self.get_state(), reward, self.is_game_over()

    def get_state(self):
        return [cell for row in self.board for cell in row]


if __name__ == '__main__':
    game = Game2048()
    game.print_board()
    while not game.is_game_over():
        print("UP")
        state, rew, done = game.step(0)
        print('state ', state, 'reward ', rew, 'done', done)
        print("\nLEFT")
        state, rew, done = game.step(2)
        print('\nstate ', state, 'reward ', rew, 'done', done)
        print("\nRIGHT")
        state, rew, done = game.step(3)
        print('\nstate ', state, 'reward ', rew, 'done', done)
        print("\nDOWN")
        state, rew, done = game.step(1)
        print('\nstate ', state, 'reward ', rew, 'done', done)

    print("Game over")
