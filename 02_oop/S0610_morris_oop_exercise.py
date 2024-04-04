class Miner:
    def __init__(self, name):
        self.name = name
        self.sleepiness = 0
        self.thirst = 0
        self.hunger = 0
        self.whisky = 0
        self.gold = 0
        self.roundnumber = 0
        self.miner = {"miner": self.name, "round": self.roundnumber, "sleepiness": self.sleepiness, "thirst": self.thirst, "hunger": self.hunger, "whisky": self.whisky, "gold": self.gold}

    def _sleep(self):
        self.miner["sleepiness"] -= 10
        self.miner["thirst"] += 1
        self.miner["hunger"] += 1


    def _drink(self):
        self.miner["sleepiness"] += 5
        self.miner["thirst"] -= 15
        self.miner["hunger"] -= 1
        self.miner["whisky"] -= 1


    def _eat(self):
        self.miner["sleepiness"] += 5
        self.miner["thirst"] -= 5
        self.miner["hunger"] -= 20
        self.miner["gold"] -= 2

    def _mine(self):
        self.miner["sleepiness"] += 5
        self.miner["thirst"] += 5
        self.miner["hunger"] += 5
        self.miner["gold"] += 5

    def _buy_whisky(self):
        self.miner["sleepiness"] += 5
        self.miner["thirst"] += 1
        self.miner["hunger"] += 1
        self.miner["whisky"] += 1
        self.miner["gold"] -= 1

    def dead(self):
        if self.miner["sleepiness"] > 100 or self.miner["thirst"] > 100 or self.miner["hunger"] > 100:
            return True
        else:
            return False

    def getTurn(self):
        return self.miner["round"]


    @property
    def round(self):
        if self.miner["sleepiness"] >= 95:
            self._sleep()
        elif self.miner["thirst"] >= 93:
            self._buy_whisky()
            self.miner["round"] += 1
            if not self.dead():
                print(self.miner)
                self._drink()
            else:
                print(self.miner)
                quit()
        elif self.miner["hunger"] >= 94:
            self._eat()
            self.miner["round"] += 1
        else:
            self._mine()
        self.miner["round"] += 1

        print(self.miner)
        return self.miner["round"]

def the_miner():
    morris = Miner("Morris")
    turn = morris.getTurn()
    while not morris.dead() == True and turn < 1000:
        morris.round
        turn = morris.getTurn()



the_miner()

