"""Opgave "Turtle Hunt":

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Spillet:
    Denne øvelse er et spil for 2 spillere.
    3 skildpadder (jægere) forsøger at fange en anden skildpadde (bytte) så hurtigt som muligt.
    Den ene spiller styrer byttet, den anden spiller styrer jægerne. Derefter bytter spillerne roller.
    I hver tur bestemmer spillerne, hvor mange grader deres skildpadde(r) roterer. Dette er spillerens eneste opgave.
    Byttet får et point for hver tur, før det bliver fanget.
    Hvis byttet stadig er på flugt efter MAX_TURNS omgange, fordobles pointene, og jagten slutter.


Koden til spillet er allerede skrevet i S1530_turtle_hunt_main.py og S1520_turtle_hunt_service.py. Du må ikke ændre disse filer.

Din opgave er at få skildpadderne til at rotere smartere.

Kopier alle 3 turtle_hunt-filer til din egen løsningsmappe.
Skriv din løsning ind i din kopi af S1510_turtle_hunt_classes_constants.py.

Filstruktur:
    Koden er opdelt i 3 filer for at gøre det klart, hvilken del af koden
    du skal ændre, og hvilken del af koden du skal lade være uændret.

    S1530_turtle_hunt_main.py:
        Dette er hovedprogrammet.
        Du må ikke foretage ændringer heri.
        Kør det for at starte spillet.

    S1520_turtle_hunt_service.py:
        Indeholder nogle servicefunktioner, som vil være nyttige for dig.
        Du må ikke foretage ændringer heri.

    Denne fil (S1510_turtle_hunt_classes_constants.py):
        Alt din programmering til denne øvelse foregår i denne fil.

Delopgaver:
Trin 1:
    Kig på programkoden.
    Du behøver ikke at forstå alle detaljer i hovedprogrammet.
    Forstå, hvordan de tre filer importerer hinandens kode med "import".
    Forstå, hvornår og hvordan metoderne rotate_prey() og rotate_hunt() bruges.
    Fra nu af foregår al din programmering til denne øvelse i denne fil her.

Trin 2:
    Ændr navnet på klassen PlayerName1 i den første linje i dens klassedefinition til dit personlige klasses navn.
    Nederst i denne fil skal du sætte variablerne class1 og class2 til dit personlige klasses navn.

Trin 3:
    I din personlige klasse skal du gøre metoderne rotate_prey og rotate_hunter smartere.
    Eventuelt vil du også tilføje nogle attributter og/eller metoder til din klasse.
    Du må dog ikke manipulere (f.eks. flytte) skildpadderne med din kode.
    Køretiden for dine metoder rotate_prey og rotate_hunter skal være mindre end 0,1 sekunder pr. iteration.

Trin 4:
    Find en sparringspartner i din studiegruppe.
    Som med alt andet skal du bede din lærer om hjælp, hvis det er nødvendigt.
    I din kode skal du erstatte hele klassen PlayerName2 med din sparringspartners klasse.
    Nederst i denne fil indstiller du variablen class2 til din sparringpartners klasses navn.
    Lad de 2 klasser kæmpe og lær af resultaterne for at forbedre din kode.
    Gentag dette trin indtil du er tilfreds :-)

Trin 5:
    Når dit program er færdigt, skal du skubbe det til dit github-repository.
    Send derefter denne Teams-besked til din lærer: <filename> done
    Derefter fortsætter du med den næste fil.

Senere:
    Når alle er færdige med denne øvelse, afholder vi en turnering
    for at finde ud af, hvem der har programmeret de klogeste skildpadder :)

Kun hvis du er nysgerrig og elsker detaljer:
    Her er den fulde dokumentation for skildpaddegrafikken:
    https://docs.python.org/3.3/library/turtle.html"""

import math
import turtle  # this imports a library called "turtle". A library is (someone else's) python code, that you can use in your own program.
import random
from S1520_turtle_hunt_service import distance, direction


class Bejap(turtle.Turtle):

    def __init__(self):
        super().__init__()  # Here, this is equivalent to turtle.Turtle.__init__(self)
        self.orientation = 0  # used to keep track of the turtle's current orientation (the direction it is heading)

    def rotate_prey(self, positions):
        prey_pos = positions[0]
        hunters_pos = positions[1:]


        closest_hunter_pos = self.find_closest_hunter(prey_pos, hunters_pos)

        # Calculate the direction away from the closest hunter
        direction_away_from_hunter = self.calculate_direction_away_from_hunter(prey_pos, closest_hunter_pos)

        # Calculate the rotation angle to face away from the closest hunter
        degree = direction_away_from_hunter - self.orientation  + random.uniform(-5, 5)

        # Update orientation and return rotation
        self.orientation = direction_away_from_hunter
        return degree

    def find_closest_hunter(self, prey_pos, hunters_pos):
        # Calculate distances from prey to each hunter
        distances_to_hunters = [distance(prey_pos, hunter_pos) for hunter_pos in hunters_pos]

        # Find the index of the closest hunter
        closest_hunter_index = distances_to_hunters.index(min(distances_to_hunters))

        # Return the position of the closest hunter
        return hunters_pos[closest_hunter_index]

    def calculate_direction_away_from_hunter(self, prey_pos, hunter_pos):
        # Calculate the direction from the prey to the hunter
        direction_to_hunter = direction(prey_pos, hunter_pos)

        # Calculate the opposite direction
        opposite_direction = (direction_to_hunter + 180) % 360

        return opposite_direction

    def rotate_hunter(self, positions):
        hunter_pos = self.position()
        prey_pos = positions[0]

        # Calculate angle to the prey
        angle_to_prey = direction(hunter_pos, prey_pos)

        # Find the direction that minimizes the angle to the prey
        optimal_direction = self.find_optimal_direction(angle_to_prey)

        # Calculate the rotation angle to face the optimal direction
        rotation_angle = optimal_direction - self.orientation

        # Update orientation and return rotation
        self.orientation = optimal_direction
        return rotation_angle

    def find_optimal_direction(self, angle_to_prey):
        # Define a range of angles
        num_angles = 36  # Number of angles to consider
        angle_step = 360 / num_angles
        angles = [i * angle_step for i in range(num_angles)]

        # Calculate absolute differences between angle to prey and each angle
        angle_differences = [abs(angle - angle_to_prey) for angle in angles]

        # Find the angle that minimizes the absolute difference
        optimal_direction_index = angle_differences.index(min(angle_differences))
        optimal_direction = angles[optimal_direction_index]

        return optimal_direction


#  Insert the code of your sparring partner's turtle class here:
#
#
#
#


# change these global constants only for debugging purposes:
MAX_TURNS = 100       # Maximum number of turns in a hunt.                           In competition: probably 200.
ROUNDS = 1            # Each player plays the prey this often.                       In competition: probably 10.
STEP_SIZE = 3         # Distance each turtle moves in one turn.                      In competition: probably 3.
SPEED = 0             # Fastest: 10, slowest: 1, max speed: 0.                       In competition: probably 0.
CAUGHT_DISTANCE = 10  # Hunt is over, when a hunter is nearer to the prey than that. In competition: probably 10.


random.seed(2)  # use seed() if you want reproducible random numbers for debugging purposes. You may change the argument of seed().


class1 = Bejap  # (red prey) Replace PlayerName1 by your own class name here.
class2 = Bejap  # (green prey) For testing your code, replace PlayerName1 by your own class name here. Later replace this by your sparring partner's class name.
