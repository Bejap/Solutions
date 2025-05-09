import Whist as whist

def main(n):
    players = ["Tommy", "Bobby", "Preo", "Mark"]
    game = whist.Whist(players)
    results = []
    to_list = [0, 0]

    for _ in range(n):
        to_list = [0, 0]
        while max(to_list) <= 7:
            result = game.play_game()
            if result[0] == 0:
                to_list[0] += result[1]
            else:
                to_list[1] += result[1]

            print(f"Team 1: {to_list[0]}\n Team 2: {to_list[1]}")

        results.append(tuple(to_list))

    print(f"\nGame results: {to_list}")

main(2)
