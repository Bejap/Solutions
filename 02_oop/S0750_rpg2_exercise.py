"""opgave: Objektorienteret rollespil, afsnit 2 :

Som altid skal du læse hele øvelsesbeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Byg videre på din løsning af afsnit 1.

I hver omgang bruger en figur en af sine evner (metoder). Derefter er det den anden figurs tur.
Det er op til dig, hvordan dit program i hver tur beslutter, hvilken evne der skal bruges.
Beslutningen kan f.eks. være baseret på tilfældighed eller på en smart strategi





Hvis du går i stå, kan du spørge google, de andre elever eller læreren (i denne rækkefølge).

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-besked til din lærer: <filename> done
Fortsæt derefter med den næste fil."""
import random
class Character:

    def __init__(self, name, max_health, attackpower):
        self.name = name
        self.max_health = max_health
        self.attackpower = attackpower
        self._current_health = self.max_health

    def __repr__(self):
        return f'''{self.name} the {self.__class__.__name__} has {self.max_health} health and an attackpower of {self.attackpower}.
        Current health is {round(self._current_health, 2)}\n'''

    # Del 3:
    #     Hver gang en figur bruger en af sine evner, skal du tilføje noget tilfældighed til den anvendte evne.

    def hit(self, other):
        if type(other):
            other.get_hit(round(self.attackpower * (random.uniform(0.2, 2)), 2))

    def get_healed(self, healpower):
        self._current_health += (round(healpower * random.uniform(0.2, 2), 2))

    def get_hit(self, opponentdmg):
        self._current_health -= opponentdmg
        print(f"hit was successfull and took the health of {self.name} to \n\t {round(self._current_health,2)}\n")

    def dead(self):
        if self._current_health <= 0:
            print(f'{self.name} has died')
            return True

    def health(self):
        return self._current_health



class Healer(Character):
    def __init__(self, name, max_health, healpower):
        super().__init__(name, max_health, attackpower=0)
        self.name = name
        self.max_health = max_health
        self.attackpower = 0
        self.healpower = healpower

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n        The healer has {self.healpower} healpower\n'''

    def heal(self, other):
        other.get_healed(self.healpower)

# Del 1:
#     Opfind to nye klasser, som arver fra klassen Character. For eksempel Hunter og Magician.
#     Dine nye klasser skal have deres egne ekstra metoder og/eller attributter.
#     Måske overskriver de også metoder eller attributter fra klassen Character.

class Magician(Character):
    def __init__(self, name, max_health, healpower, attackpower, magicpower):
        super().__init__(name, max_health, attackpower)
        self.name = name
        self.max_health = max_health
        self.healpower = healpower
        self.attackpower = attackpower
        self.magicpower = magicpower

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n       The magician has {self.magicpower} magicpower and {self.healpower} healpower\n'''

    def heal(self, other):
        other.get_healed(self.healpower)

    def hit(self, other):
        other.get_hit(self.attackpower)

    def meditate(self):
        self.magicpower += 10 + (0.1 * self.magicpower)
        self.attackpower += self.magicpower * (0.001 * self.magicpower + 0.01)
        self.healpower += self.magicpower * (0.001 * self.magicpower + 0.01)
        print("The meditation was successfull, and the Magician now has ", round(self.magicpower,2), "magicpower, ", round(self.healpower,2),  "healpower and ", round(self.attackpower, 2), "attackpower\n")

class Horseman(Character):
    def __init__(self, name, max_health, attackpower, horsehealth):
        super().__init__(name, max_health, attackpower)
        self.name = name
        self.max_health = max_health
        self.attackpower = attackpower
        self.horsehealth = horsehealth
        self._current_horse_health = horsehealth

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n        The Horse max health is {self.horsehealth}
        The Horse current health is {self._current_horse_health}\n'''

    def get_hit(self, opponentdmg):
        if self._current_horse_health <= 0:
            self._current_health -= opponentdmg
            print(f"The hit was successfull, and the hit took the {self.name} health to \n\t", self._current_health, "\n")
        else:
            self._current_horse_health -= opponentdmg * 2
            print("The hit was successfull, and the hit took the horse health to \n\t", self._current_horse_health, "\n")

    def hit(self, opponent):
        if self._current_horse_health <= 0:
            opponent.get_hit(round(self.attackpower * (random.uniform(0.5, 2)), 2))
        else:
            opponent.get_hit(round(self.attackpower * 2 * (random.uniform(0.5, 2)), 2))

# Del 2:
#     Lad i hovedprogrammet objekter af dine nye klasser (dvs. rollespilfigurer) kæmpe mod hinanden,
#     indtil den ene figur er død. Udskriv, hvad der sker under kampen.


def fight():

    t = Character("Oprah", 600, 120)
    y = Healer("Tom", 450, 100)
    m = Magician("Tarija", 1200, 20, 30, 100)
    h = Horseman("Ika", 750, 70, 450)
    while not any([t.dead(), y.dead(), m.dead(), h.dead()]):
        for attacker in [t, y, m, h]:
            opponent = random.choice([c for c in [t, m, h] if c != attacker and not c.dead()])
            attacker.hit(opponent)
            if opponent.dead():
                return opponent

def main():
    twin = ywin = mwin = hwin = 0
    for i in range(100):
        winner = fight()
        if winner is not None:
            if not Character == type(winner):
                twin += 1
            if not Magician == type(winner):
                mwin += 1
            if not Horseman == type(winner):
                hwin += 1
    print("Character wins: {0}, Magician wins: {1}, Horseman wins: {2}".format(twin, mwin, hwin))

main()

# Del 4:
#     Lad dine figurer kæmpe mod hinanden 100 gange.
#     Hold styr på resultaterne.
#     Prøv at afbalancere dine figurers evner på en sådan måde, at hver figur vinder ca. halvdelen af kampene.



