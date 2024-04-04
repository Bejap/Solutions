

def morris_the_man():
    turn = 0
    sleepiness = 0
    thirst = 0
    hunger = 0
    gold = 0
    whisky = 0
    morris = {"turn": turn, "sleepiness": sleepiness, "thirst": thirst, "hunger": hunger, "gold": gold, "whisky": whisky}
    while not dead(morris) == True and morris["turn"] < 1000:
        if morris["sleepiness"] >= 95:
            sleep(morris)
        elif morris["thirst"] >= 94:
            buy_whisky(morris)
            morris["turn"] += 1
            print(morris)
            drink(morris)
        elif morris["hunger"] >= 94:
            eat(morris)
        else:
            mine(morris)
        morris["turn"] += 1
        print(morris)



def sleep(morris):
    morris["sleepiness"] -= 10
    morris["thirst"] += 1
    morris["hunger"] += 1
    return morris
def buy_whisky(morris):
    morris["sleepiness"] += 5
    morris["thirst"] += 1
    morris["hunger"] +=1
    morris["gold"] -=1
    morris["whisky"] +=1
    return morris

def drink(morris):
    morris["sleepiness"] += 5
    morris["thirst"] -= 15
    morris["hunger"] -= 1
    morris["whisky"] -=1
    return morris
def eat(morris):
    morris["sleepiness"] +=5
    morris["thirst"] -=5
    morris["hunger"] -=20
    morris["gold"] -= 2
    return morris

def mine(morris):
    morris["sleepiness"] +=5
    morris["thirst"] +=5
    morris["hunger"] +=5
    morris["gold"] += 5
    return morris

def dead(morris):
    if morris["sleepiness"] <= 100 or morris["thirst"] <= 100 or morris["hunger"] <= 100:
        return False
    else:
        return True

morris_the_man()