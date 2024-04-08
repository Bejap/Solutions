"""opgave: Objektorienteret rollespil, afsnit 2 :

Som altid skal du læse hele øvelsesbeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Byg videre på din løsning af afsnit 1.

Del 1:
    Opfind to nye klasser, som arver fra klassen Character. For eksempel Hunter og Magician.
    Dine nye klasser skal have deres egne ekstra metoder og/eller attributter.
    Måske overskriver de også metoder eller attributter fra klassen Character.

Del 2:
    Lad i hovedprogrammet objekter af dine nye klasser (dvs. rollespilfigurer) kæmpe mod hinanden,
    indtil den ene figur er død. Udskriv, hvad der sker under kampen.

I hver omgang bruger en figur en af sine evner (metoder). Derefter er det den anden figurs tur.
Det er op til dig, hvordan dit program i hver tur beslutter, hvilken evne der skal bruges.
Beslutningen kan f.eks. være baseret på tilfældighed eller på en smart strategi

Del 3:
    Hver gang en figur bruger en af sine evner, skal du tilføje noget tilfældighed til den anvendte evne.

Del 4:
    Lad dine figurer kæmpe mod hinanden 100 gange.
    Hold styr på resultaterne.
    Prøv at afbalancere dine figurers evner på en sådan måde, at hver figur vinder ca. halvdelen af kampene.

Hvis du går i stå, kan du spørge google, de andre elever eller læreren (i denne rækkefølge).

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-besked til din lærer: <filename> done
Fortsæt derefter med den næste fil."""

class Character:

    def __init__(self, name, max_health, attackpower):
        self.name = name
        self.max_health = max_health
        self.attackpower = attackpower
        self._current_health = self.max_health



    def __repr__(self):
        return f'''{self.name} has {self.max_health} and an attackpower of {self.attackpower}.
        Current health is {self._current_health}'''



    def hit(self, other):
        if type(other) == Horseman:
            other.get_hit(self.attackpower)
        else:
            other._current_health -= self.attackpower
            print("The hit was successfull, and the hit took the opponents health to ", other._current_health, "\n")

    def get_healed(self, healpower):
        self._current_health += healpower

    def get_hit(self, opponentdmg):
        self._current_health -= opponentdmg

    def dead(self):
        if self._current_health <= 0:
            return True
        else:
            return False



class Healer(Character):
    def __init__(self, name, max_health, healpower):
        self.name = name
        self.max_health = max_health
        self.attackpower = 0
        self.healpower = healpower
        self._current_health = self.max_health


    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n        The healer has {self.healpower} healpower\n'''

    def heal(self, other):
        other.get_healed(self.healpower)

    def get_healed(self, healpower):
        self._current_health += healpower

    def get_hit(self, opponentdmg):
        self._current_health -= opponentdmg


class Magician(Character):
    def __init__(self, name, max_health, healpower, attackpower, magicpower):
        self.name = name
        self.max_health = max_health
        self.healpower = healpower
        self.attackpower = attackpower
        self.magicpower = magicpower
        self._current_health = self.max_health

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n       The magician has {self.magicpower} and {self.healpower} healpower\n'''

    def heal(self, other):
        other.get_healed(self.healpower)


    def hit(self, other):
        other.get_hit(self.attackpower)

    def get_healed(self, healpower):
        self._current_health += healpower

    def get_hit(self, opponentdmg):
        self._current_health -= opponentdmg


    def meditate(self):
        self.magicpower += 10 + (0.1 * self.magicpower)
        self.attackpower += self.magicpower * (0.001 * self.magicpower + 0.01)
        self.healpower += self.magicpower * (0.001 * self.magicpower + 0.01)
        print("The meditation was successfull, and the Magician now has ", self.magicpower, "magicpower, ", self.healpower, "healpower and ", self.attackpower, "attackpower\n" )

class Horseman(Character):
    def __init__(self, name, max_health, attackpower, horsehealth):
        self.name = name
        self.max_health = max_health
        self.attackpower = attackpower
        self.horsehealth = horsehealth
        self._current_health = max_health
        self._current_horse_health = horsehealth

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n        The Horse has health, {self._current_horse_health}
        The Horse current health is {self._current_horse_health}'''
    def get_hit(self, opponentdmg):
        if self._current_horse_health <= 0:
            self._current_health -= opponentdmg
            print("The hit was successfull, and the hit took the horsemans health to ", self._current_health, "\n")
        else:
            self._current_horse_health -= opponentdmg * 2
            print("The hit was successfull, and the hit took the horse health to ", self._current_horse_health, "\n")

    def hit(self, opponent):
        if self._current_horse_health <= 0:
            opponent.get_hit(self.attackpower)
        else:
            opponent.get_hit(self.attackpower * 2)

    def get_healed(self, healpower):
        self._current_health += healpower


t = Character("oprah", 600, 120)
y = Healer("Tom", 450, 100)
m = Magician("hom", 1200, 20, 30, 100)
h = Horseman("ika", 750, 70, 450)

print(t)
print(y)
print(m)
print(h)


for i in range(10):
    if t.dead() or h.dead():
        break
    t.hit(h)
    h.hit(t)


