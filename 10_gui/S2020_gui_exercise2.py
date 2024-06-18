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
    def empty_entry():  # Delete text in the entry box
        print("button_1 was pressed")
        entry_1.delete(0, END)
        entry_2.delete(0, END)
        entry_3.delete(0, END)
        entry_4.delete(0, END)

    win = Tk()
    win.title('my first GUI')
    win.geometry('493x179')

    # create frame
    label_frame = LabelFrame(win, text='Container', width=460, height=200)
    label_frame.grid(row=0, column=0, padx=10, pady=5, sticky=N)
    frame_1 = Frame(label_frame)
    frame_1.grid(row=0, column=0, padx=20, pady=15, sticky=N)

    id = Label(frame_1, text='Id')
    id.grid(row=0, column=0, pady=5, padx=(15, 25))
    weight = Label(frame_1, text='Weight')
    weight.grid(row=0, column=1, pady=5, padx=(0, 20))
    destination = Label(frame_1, text='Destination')
    destination.grid(row=0, column=2, pady=5, padx=(5, 20))
    weather = Label(frame_1, text='Weather')
    weather.grid(row=0, column=3, pady=5, padx=(0, 20))
    entry_1 = Entry(frame_1, width=4, justify='left')
    entry_1.grid(row=1, column=0, pady=5, padx=0)
    entry_1.insert(0, '')
    entry_2 = Entry(frame_1, width=8, justify='left')
    entry_2.grid(row=1, column=1, pady=5, padx=(0, 20))
    entry_2.insert(0, '')
    entry_3 = Entry(frame_1, width=18, justify='left')
    entry_3.grid(row=1, column=2, pady=5, padx=(0, 20))
    entry_3.insert(0, '')
    entry_4 = Entry(frame_1, width=15, justify='left')
    entry_4.grid(row=1, column=3, pady=5)
    entry_4.insert(0, '')

    frame_2 = Frame(frame_1)
    frame_2.grid(row=2, columnspan=4, pady=15)

    big_button = Button(frame_2, text='Create', width=5)
    big_button.grid(row=0, column=0, padx=5)

    update_button = Button(frame_2, text='Update', width=6)
    update_button.grid(row=0, column=1, padx=5)

    del_button = Button(frame_2, text='Delete', width=6)
    del_button.grid(row=0, column=2, padx=5)

    clear_en_button = Button(frame_2, text='Clear Entry Boxes', width=13, command=empty_entry)
    clear_en_button.grid(row=0, column=3, padx=5)


    win.mainloop()


if __name__ == '__main__':
    main()
