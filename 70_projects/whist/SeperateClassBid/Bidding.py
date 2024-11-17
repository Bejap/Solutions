class Player:
    def __init__(self, name):
        self.name = name
        self.bid = None  # None indicates no bid yet
        self.passed = False  # Tracks if the player has passed

    def make_bid(self, current_highest_bid):
        """
        Simulates a player's bid or allows them to pass.
        """
        if self.passed:
            print(f"{self.name} has already passed.")
            return self.bid

        while True:
            action = int(input(f"{self.name}, do you want to bid or pass? (1/0): "))
            if not action:
                self.passed = True
                print(f"{self.name} has passed.")
                return None
            elif action:
                try:
                    bid = int(input(f"Enter your bid (must be higher than {current_highest_bid}): "))
                    if bid > current_highest_bid and bid <= 13:
                        self.bid = bid
                        return bid
                    else:
                        print(f"Bid must be higher than the current highest bid ({current_highest_bid}, but max 13).")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("Invalid action. Please type 'bid' or 'pass'.")

class BiddingSystem:
    def __init__(self, players):
        self.players = players
        self.highest_bid = 6
        self.highest_bidder = None
        self.active_players = len(players)

    def start_bidding(self):
        print("Starting the bidding process...")

        while self.active_players > 1:  # Continue until only one player hasn't passed
            for player in self.players:
                if player.passed:
                    continue

                bid = player.make_bid(self.highest_bid)
                if bid is not None and bid > self.highest_bid:
                    self.highest_bid = bid
                    self.highest_bidder = player

                # Check if all players except one have passed
                self.active_players = sum(not p.passed for p in self.players)
                if self.active_players <= 1:
                    break

        if self.highest_bidder:
            print(f"Bidding ended. The winner is {self.highest_bidder.name} with a bid of {self.highest_bid}.")
        else:
            print("No bids were placed. No winner.")

# Example Usage
if __name__ == "__main__":
    players = [Player("Player 1"), Player("Player 2"), Player("Player 3"), Player("Player 4")]
    bidding_system = BiddingSystem(players)
    bidding_system.start_bidding()
