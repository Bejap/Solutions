""" Opgave "GUI step 2":

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Bruge det, du har lært i GUI-eksempelfilerne, og byg den GUI, der er afbildet i images/gui_2020.png

Genbrug din kode fra "GUI step 1".

GUI-strukturen bør være som følger:
    main window
        labelframe
            frame
                labels and entries
            frame
                buttons

Funktionalitet:
    Klik på knappen "clear entry boxes" sletter teksten i alle indtastningsfelter (entries).

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""


from tkinter import *

def main():
    win = Tk()
    win.title('my first GUI')
    win.geometry('493x179')

    # create frame
    label_frame = LabelFrame(win, text='Container')
    label_frame.grid(row=0, column=0, padx=10, pady=5, sticky=N)

    frame_1 = Frame(label_frame)
    frame_1.grid(row=0, column=0, padx=15, pady=15, sticky=N)

    id = Label(frame_1, text='Id')
    id.grid(row=0, column=0, padx=25)

    weight = Label(frame_1, text='Weight')
    weight.grid(row=0, column=1, padx=5)

    destination = Label(frame_1, text='Destination')
    destination.grid(row=0, column=2, padx=35)

    weather = Label(frame_1, text='Weather')
    weather.grid(row=0, column=3, padx=30)

    entry_1 = Entry(frame_1, width=4, justify='right')
    entry_1.grid(row=1, column=0, padx=0, pady=5)

    entry_2 = Entry(frame_1, width=8, justify='right')
    entry_2.grid(row=1, column=1)

    entry_3 = Entry(frame_1, width=18, justify='right')
    entry_3.grid(row=1, column=2)

    entry_4 = Entry(frame_1, width=15, justify='right')
    entry_4.grid(row=1, column=3)

    frame_2 = Frame(label_frame)
    frame_2.grid(row=1, column=0)

    big_buttom = Button(frame_2, text='Create')
    big_buttom.grid(row=0, column=0, pady=10)


    win.mainloop()


if __name__ == '__main__':
    main()
