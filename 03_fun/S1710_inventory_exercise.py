"""Opgave "The inventory sequence"

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Denne øvelse er en valgfri udfordring for de fremragende programmører blandt jer.
Du behøver absolut ikke at løse denne øvelse for at fortsætte med succes.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Del 1:
    Se de første 3 minutter af denne video:
    https://www.youtube.com/watch?v=rBU9E-ZOZAI

Del 2:
    Skriv en funktion inventory(), som producerer de tal, der er vist i videoen.
    Funktionen accepterer en parameter, der definerer, hvor mange talrækker der skal produceres.
    Funktionen udskriver tallene i hver række.

    Du vil sandsynligvis ønske at definere en funktion count_number(), som tæller, hvor ofte
    et bestemt antal optræder i den aktuelle talrække.

Del 3:
    I hovedprogrammet kalder du inventory() med fx 6 som argument.

Hvis du ikke har nogen idé om, hvordan du skal begynde, kan du kigge på løsningen
i S1720_inventory_solution.py

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""

def count_number(y, i):  # the function accepts two parameters, y is the list that is being checked, i is the value to check for.
    count = 0
    for k in y:  # loops through the list
        if k == i:  # checks if an element of y is the value to check for
            count += 1  # counting the amount of times it occurs.
    return count


def inventory(t):
    list0 = []  # create the initial row
    count = 0  # creates a count for fun
    for i in range(t + 1):  # initiates a loop with iterations according to the value requested
        a_list = []  # creates the list for the elements
        for j in range(t * t):  # initiates a loop that will iterate 100 times
            c = count_number(list0, j)  # find the amount of times a certain number occours
            if j <= len(list0):  # makes sure, that the list is finite
                list0.append(c)  # extending the list, so that the elements are countable
                a_list.append(c)  # gives the elements to the printable list
            if c == 0:
                break
        print(f"{a_list}   {count}")
        count += 1


inventory(30)