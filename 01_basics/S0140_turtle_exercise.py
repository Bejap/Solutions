"""
Opgave "Tom the Turtle":

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Funktionen "demo" introducerer dig til alle de kommandoer, du skal bruge for at interagere med Tom i de følgende øvelser.

Kun hvis du er nysgerrig og elsker detaljer:
    Her er den fulde dokumentation for turtle graphics:
    https://docs.python.org/3.3/library/turtle.html

Del 1:
    Skriv en funktion "square", som accepterer en parameter "length".
    Hvis denne funktion kaldes, får skildpadden til at tegne en firkant med en sidelængde på "længde" pixels.

Del 2:
     Færdiggør funktionen "visible", som skal returnere en boolsk værdi,
     der angiver, om skildpadden befinder sig i det synlige område af skærmen.
     Brug denne funktion i de følgende dele af denne øvelse
     til at få skildpadden tilbage til skærmen, når den er vandret væk.

Del 3:
    Skriv en funktion "many_squares" med en for-loop, som kalder square gentagne gange.
    Brug denne funktion til at tegne flere firkanter af forskellig størrelse i forskellige positioner.
    Funktionen skal have nogle parametre. F.eks:
        antal: hvor mange firkanter skal der tegnes?
        størrelse: hvor store er firkanterne?
        afstand: hvor langt væk fra den sidste firkant er den næste firkant placeret?

Del 4:
    Skriv en funktion, der producerer mønstre, der ligner dette:
    https://pixabay.com/vectors/spiral-square-pattern-black-white-154465/

Del 5:
    Skriv en funktion, der producerer mønstre svarende til dette:
    https://www.101computing.net/2d-shapes-using-python-turtle/star-polygons/
    Funktionen skal have en parameter, som påvirker mønsterets form.

Del 6:
    Opret din egen funktion, der producerer et sejt mønster.
    Senere, hvis du har lyst, kan du præsentere dit mønster på storskærmen for de andre.

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil.
"""
def visible(turtle_name):
    if 350 < Tom.xcor():
        return True
    elif -350 > Tom.xcor():
        return True
    elif 350 < Tom.ycor():
        return True
    elif -350 > Tom.ycor():
        return True
    else:
        return False





import turtle

Tom = turtle.Turtle()
Tom.speed(5)




def square_pattern(spiral_turns, space, length, drawsize):
    Tom.right(90)
    Tom.pensize(drawsize)
    for i in range(spiral_turns):
        Tom.forward(length)
        Tom.left(90)
        length += space



        if visible(Tom) == True:
            Tom.penup()
            Tom.home()



    turtle.done()

#square_pattern(35, 20, 20, 2)

def starforming(tacks, size):

    Tom.speed(10)
    if tacks % 2 == 0:
        for i in range(tacks):
            tacks_degree =  (1080/tacks)
            Tom.right(tacks_degree)
            Tom.forward(size)
    else:
        for i in range(tacks):
            tacks_degree = (180/tacks) + 180
            Tom.right(tacks_degree)
            Tom.forward(size)

    Tom.penup()
    Tom.forward(30)
    turtle.done()



starforming(12, 50)


