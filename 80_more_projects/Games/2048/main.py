from game2048 import Game2048
import keyboard
import time


def main():
    game = Game2048()
    game.print_board()

    print("\nUse W (↑), A (←), S (↓), D (→) to play. Press Q to quit.")

    while not game.is_game_over():
        if keyboard.is_pressed("w"):
            game.move_up(True)
            time.sleep(0.15)
        elif keyboard.is_pressed("s"):
            game.move_down(True)
            time.sleep(0.15)
        elif keyboard.is_pressed("a"):
            game.move_left()
            time.sleep(0.15)
        elif keyboard.is_pressed("d"):
            game.move_right()
            time.sleep(0.15)
        elif keyboard.is_pressed("r"):
            print("\nGame resetting")
            time.sleep(0.5)
            game.reset()
        elif keyboard.is_pressed("q"):
            print("Quitting the game.")
            break

    if game.is_game_over():
        print("\nGame Over!")


if __name__ == "__main__":
    main()
