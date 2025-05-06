from game2048 import Game2048

game = Game2048()

def reset():
    game.board = [
        [2, 0, 0, 2],
        [4, 4, 4, 4],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    return game.board


def test_move_right(board):
    game.move_right(show=False, add_tile=False)
    expected_board_right = [
        [0, 0, 0, 4],
        [0, 0, 8, 8],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]

    assert board == expected_board_right


def test_move_left(board):
    game.move_left(show=False, add_tile=False)
    expected_board_left = [
        [4, 0, 0, 0],
        [8, 8, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert board == expected_board_left


def test_move_up(board):
    game.move_up(False)
    expected_board_up = \
        [[2, 4, 4, 2],
         [4, 0, 2, 4],
         [0, 0, 0, 0],
         [0, 0, 0, 0], ]

    assert board == expected_board_up


def test_move_down(board):
    game.move_down(False)
    expected_board_down = \
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [2, 0, 4, 2],
         [4, 4, 2, 4], ]

    assert board == expected_board_down


game.board = reset()
test_move_right(game.board)
game.board = reset()
test_move_left(game.board)
game.board = reset()
test_move_up(game.board)
game.board = reset()
test_move_down(game.board)
