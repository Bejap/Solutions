import Whist as whist

def main(n):
    players = ["Tommy", "Bobby", "Preo", "Mark"]
    game = whist.Whist(players)
    results = []

    for _ in range(n):
        to_list = [0, 0]
        i = 0
        while max(to_list) <= 7:
            result = game.play_game(i)
            if result[0] == 0:
                to_list[0] += result[1]
            else:
                to_list[1] += result[1]

            print(result[1], "Team 1: ", to_list[0],"Team 2:", to_list[1])
            i += 1

        results.append(tuple(to_list))

    print(f"\nGame results: {to_list}")

main(1)
