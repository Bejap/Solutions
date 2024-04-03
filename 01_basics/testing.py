import turtle
import random

def visible(Tom):
    if Tom.position()[0] > 200 or Tom.position()[0] < -200:
        Tom.right(13)
        return True

    elif Tom.position()[1] > 200 or Tom.position()[1] < -200:
        Tom.right(13)

        return True
    else:
        return False

Tom = turtle.Turtle()
Tom.speed(50)

def movement():
    for i in range(100):
        Tom.forward(random.random() * 50)
        if visible(Tom):

            Tom.goto(0,0)
    turtle.done()
movement()

