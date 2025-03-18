import Deck as dck
import Player as plr


class Whist:
    def __init__(self, player_names: list[str]):
        # Assign players to teams
        self.players = [plr.Player(name, is_human=False, team=i % 2) for i, name in enumerate(player_names)]
        self.count = 0
        self.trump_cycle = ["Hearts", "Spades", "Diamonds", "Clubs", None]
        self.team_scores = {0: 0, 1: 0}


    def score_table(self, winner, points):
        score_list = []
        score_list.append((winner, points))
        return score_list


    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = 13
        self.count += 1

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)
            player.sort_hand()

    def play_trick(self, leading_player_index: int):
        trick_cards = []
        lead_suit = None
        winning_card = None
        winning_player = None

        print(f"\n{'=' * 50}")
        print(f"--- New Trick ---")
        print(f"Trump suit: {self.trump}")

        hand = self.players[leading_player_index].hand


        for i in range(len(self.players)):
            current_player = self.players[(leading_player_index + i) % len(self.players)]
            print(f"\n{current_player.name}'s turn:")

            playable_cards = [card for card in current_player.hand if card.suit == lead_suit] if lead_suit else current_player.hand
            if not playable_cards:
                playable_cards = current_player.hand

            played_card = current_player.play_card(playable_cards, self.trump, lead_suit, trick_cards)
            trick_cards.append(played_card)

            for player in self.players:
                player.cards_seen.add(str(played_card))

            if not lead_suit:
                lead_suit = played_card.suit
                print(f"Lead suit: {lead_suit}")

            if not winning_card:
                winning_card = played_card
                winning_player = current_player
            else:
                if (played_card.suit == winning_card.suit and played_card.rank_value > winning_card.rank_value) or \
                        (played_card.suit == self.trump and winning_card.suit != self.trump):
                    winning_card = played_card
                    winning_player = current_player

        winning_player.tricks_won += 1
        self.team_scores[winning_player.team] += 1 # need something else here
        print(f"\n{winning_player.name} wins the trick for Team {winning_player.team + 1}!")
        return self.players.index(winning_player)

    def __play_round(self, ):

        print(f"\nRound starts! Trump suit: {self.trump}")
        leading_player_index = 0
        total_tricks = len(self.players[0].hand)

        for round_num in range(total_tricks):
            print(f"\nTricks {round_num + 1}/{total_tricks}")
            leading_player_index = self.play_trick(leading_player_index)

            print("\nCurrent score:")
            for player in self.players:
                print(f"{player.name}: {player.tricks_won}/{round_num + 1} tricks ")

        winning_team = max(self.team_scores, key=self.team_scores.get)
        winning_score = self.team_scores[winning_team] - 6
        score = self.score_table(winning_team, winning_score)
        print(f"\nTeam {winning_team + 1} wins the round with a score of {winning_score}!")
        print(score)

        return winning_score


    def play_game(self):
        for game_number in range(14):
            self.trump = self.trump_cycle[game_number % len(self.trump_cycle)]
            print(f"\nStarting Game {game_number + 1} with trump suit: {self.trump if self.trump else 'No Trump'}")
            self.team_scores = {0: 0, 1: 0}  # Reset scores for the new game

            self.deck = dck.Deck()
            self.deal_cards()

            while max(self.team_scores.values()) < 7:
                self.__play_round()  # Play rounds until a team reaches 7 points

            # Declare the winning team for the game
            winning_team = max(self.team_scores, key=self.team_scores.get)
            print(f"Game {game_number + 1} winner: Team {winning_team + 1} with score: {self.team_scores[winning_team]}")



if __name__ == "__main__":
    game = Whist(['a', 'b', 'c', 'd'])
    game.play_game()




