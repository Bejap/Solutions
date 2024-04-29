"""
Opgave "GUI step 1":

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Bruge det, du har lært i GUI-eksempelfilerne, og byg den GUI, der er afbildet i images/gui_2010.png

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""

from tkinter import *

def main():
    win = Tk()
    win.title('my first GUI')
    win.geometry('125x200')

    # create frame
    frame_1 = LabelFrame(win, text='Container')
    frame_1.grid(row=0, column=0, padx=10, pady=5, sticky=N)

    my_label = Label(frame_1, text="Id")
    my_label.grid(row=1, column=4, padx=0, pady=15)

    entry_1 = Entry(frame_1, width=5, justify="right")
    entry_1.grid(row=2, column=4, padx=0, pady=10)

    create = Button(frame_1, text="Create")
    create.grid(row=3, column=4, padx=25, pady=20)


    win.mainloop()


if __name__ == '__main__':
    main()
