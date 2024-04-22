"""Opgave "Number pyramid"

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Denne øvelse er en valgfri udfordring for de fremragende programmører blandt jer.
Du behøver absolut ikke at løse denne øvelse for at fortsætte med succes.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Del 1:
    Se de første 93 sekunder af denne video: https://www.youtube.com/watch?v=NsjsLwYRW8o

Del 2:
    Skriv en funktion "pyramid", der producerer de tal, der er vist i videoen.
    Funktionen har en parameter "lines", der definerer, hvor mange talrækker der skal produceres.
    Funktionen udskriver tallene i hver række og også deres sum.

Del 3:
    I hovedprogrammet kalder du funktionen med fx 7 som argument.

Del 4:
    Tilføj en mere generel funktion pyramid2.
    Denne funktion har som andet parameter "firstline" en liste med pyramidens øverste rækkens tallene.

Del 5:
    I hovedprogrammet kalder du pyramid2 med fx 10 som det første argument
    og en liste med tal efter eget valg som andet argument.
    Afprøv forskellige lister som andet argument.

Hvis du ikke aner, hvordan du skal begynde, kan du åbne S1620_pyramid_help.py og starte derfra

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""

def next_number(row, i):
    if i == 0:
        k = row[i] + row[i+1]
    else:
        k = row[i] + row[i-1]
    return k

def old_number(row, x):
    return row[x]


def pyramid(row_amount):
    row = [1, 1]
    print(row)
    insertionrate = 1
    for i in range(row_amount - 1):
        cal = 1
        new_row = [1]
        for x in range(insertionrate):
            if x % 2 == 0:
                t = next_number(row, cal)
                new_row.append(t)
            else:
                y = old_number(row, cal)
                cal += 1
                new_row.append(y)
        new_row.append(1)
        print(new_row)
        row = new_row
        if insertionrate >= 2:
            insertionrate = 2 * insertionrate + 1
        else:
            insertionrate += 2



pyramid(6)
