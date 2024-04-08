

# Del 1:
#     Definer en klasse "Character" med attributterne "name", "max_health", "_current_health", "attackpower".
#     _current_health skal være en protected attribut, det er ikke meningen at den skal kunne ændres udefra i klassen.
# Del 2:
#     Tilføj en konstruktor (__init__), der accepterer klassens attributter som parametre.

class Character:

    def __init__(self, name, max_health, attackpower):
        self.name = name
        self.max_health = max_health
        self.attackpower = attackpower
        self._current_health = self.max_health

# Del 3:
#     Tilføj en metode til udskrivning af klasseobjekter (__repr__).
    def __repr__(self):
        return f'''{self.name} has {self.max_health} and an attackpower of {self.attackpower}.
        Current health is {self._current_health}'''


# Del 4:
#     Tilføj en metode "hit", som reducerer _current_health af en anden karakter med attackpower.
#     Eksempel: _current_health=80 og attackpower=10: et hit reducerer _current_health til 70.
#     Metoden hit må ikke ændre den private attribut _current_health i en (potentielt) fremmed klasse.
#     Definer derfor en anden metode get_hit, som reducerer _current_health for det objekt, som den tilhører, med attackpower.

    def hit(self, opponent):
        opponent.get_hit(self.attackpower)

    def get_hit(self, attackdmg):
        self._current_health -= attackdmg
        print(f'Current health is {self._current_health}')

    def get_healed(self, healer):
        self._current_health += healer
        print(f'Current health is {self._current_health}')


# Del 5:
#     Tilføj en klasse "Healer", som arver fra klassen Character.
#     En healer har attackpower=0 men den har en ekstra attribut "healpower".
class Healer(Character):
    def __init__(self, name, max_health, healpower):
        super().__init__(name, max_health, attackpower=0)
        self.name = name
        self.max_health = max_health
        self.healpower = healpower
        self._current_health = self.max_health

    def __repr__(self):
        info = super().__repr__()
        return info + f'''\n        The healer has {self.healpower} healpower\n'''

# Del 6:
#     Tilføj en metode "heal" til "Healer", som fungerer som "hit" men forbedrer sundheden med healpower.
#     For at undgå at "heal" forandrer den protected attribut "_current_health" direkte,
#     tilføj en metode get_healed til klassen Character, som fungerer lige som get_hit.
#
    def heal(self, other):
        other.get_healed(self.healpower)


player1 = Character("Bonzo", 900, 80)
player2 = Character("Kalkun", 550, 155)
player3 = Healer("Jankastíc", 250, 70)
print(player1)
print(player2)
print(player3)

player1.hit(player2)
player2.hit(player3)
player3.heal(player3)
