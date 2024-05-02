""" Opgave "GUI step 4":

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Bruge det, du har lært i GUI-eksempelfilerne, og byg den GUI, der er afbildet i images/gui_2040.png

Genbrug din kode fra "GUI step 3".

Fyld treeview'en med testdata.
Leg med farveværdierne. Find en farvekombination, som du kan lide.

Funktionalitet:
    Klik på knappen "clear entry boxes" sletter teksten i alle indtastningsfelter (entries).
    Hvis du klikker på en datarække i træoversigten, kopieres dataene i denne række til indtastningsfelterne.

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""
from tkinter import *
from tkinter import ttk
def main():
    win = Tk()
    win.title('my first GUI')
    win.geometry('491x472')
    labeled_frame = LabelFrame(win, text='Container')
    labeled_frame.grid(row=0, column=0, padx=10, pady=5, sticky=N)

    data_list = [("1", "1000", "Oslo"), ("2", "2000", "Chicago"), ("3", "3000", "Milano"), ("4", '4000', 'Amsterdam')]

    def empty_entry():
        entry_1.delete(0, END)
        entry_2.delete(0, END)
        entry_3.delete(0, END)
        entry_4.delete(0, END)

    def edit_record(event, tree):
        index_selected = tree.focus()
        values = tree.item(index_selected, 'values')
        entry_1.delete(0, END)
        entry_2.delete(0, END)
        entry_3.delete(0, END)
        entry_1.insert(0, values[0])
        entry_2.insert(0, values[1])
        entry_3.insert(0, values[2])

    def the_data(tree):
        count = 0
        for record in data_list:
            if count % 2 == 0:
                tree.insert(parent='', index='end', text='', values=record, tags='evenrow')
            else:
                tree.insert(parent='', index='end', text='', values=record, tags='oddrow')
            count += 1

    treeview_background = "#eeeeee"
    treeview_foreground = "black"
    treeview_selected = "#773333"

    evenrow = "grey81"
    oddrow = "grey87"

    style = ttk.Style()
    style.theme_use('default')
    style.configure("Treeview", background=treeview_background, foreground=treeview_foreground, rowheight=24, fieldbackground=treeview_background)
    style.map('Treeview', background=[('selected', treeview_selected)])
    frame_f_tre = Frame(labeled_frame, width=400)
    frame_f_tre.grid(row=0, column=0, pady=5)
    tree_1_scroll = Scrollbar(frame_f_tre)
    tree_1_scroll.grid(row=0, column=0, padx=10, pady=5, sticky='ns')
    tree_1 = ttk.Treeview(frame_f_tre, yscrollcommand=tree_1_scroll.set, selectmode='browse')
    tree_1.grid(row=0, column=1, padx=15, pady=6)
    tree_1_scroll.config(command=tree_1.yview)
    tree_1['columns'] = ("col1", "col2", "col3")
    tree_1.column('#0', width=0, stretch=NO)
    tree_1.column("col1", width=40, anchor=E)
    tree_1.column("col2", width=80, anchor=W)
    tree_1.column("col3", width=180, anchor=W)
    tree_1.heading("col1", text='Id', anchor=CENTER)
    tree_1.heading("col2", text='Weight', anchor=CENTER)
    tree_1.heading("col3", text='Destination', anchor=CENTER)

    tree_1.tag_configure('oddrow', background=oddrow)
    tree_1.tag_configure('evenrow', background=evenrow)
    the_data(tree_1)

    tree_1.bind("<ButtonRelease-1>", lambda event: edit_record(event, tree_1))

    frame_1 = Frame(labeled_frame)
    frame_1.grid(row=3, column=0, padx=15, pady=15, sticky=N)

    id_label = Label(frame_1, text='Id')
    id_label.grid(row=0, column=0, padx=25)

    weight = Label(frame_1, text='Weight')
    weight.grid(row=0, column=1, padx=5)

    destination = Label(frame_1, text='Destination')
    destination.grid(row=0, column=2, padx=35)

    weather = Label(frame_1, text='Weather')
    weather.grid(row=0, column=3, padx=30)

    frame_2 = Frame(labeled_frame)
    frame_2.grid(row=4, column=0)
    big_buttom = Button(frame_2, text='Create')
    big_buttom.grid(row=0, column=0, pady=10)
    up_to_date = Button(frame_2, text="Update")
    up_to_date.grid(row=0, column=1, padx=(10, 0))
    del_for_us = Button(frame_2, text='Delete')
    del_for_us.grid(row=0, column=2, padx=(10, 0))
    clearence = Button(frame_2, text='Clear Entry Boxex', command=empty_entry)
    clearence.grid(row=0, column=3, padx=(10, 0))

    entry_1 = Entry(frame_1, width=4, justify='right')
    entry_1.grid(row=1, column=0, padx=0, pady=5)

    entry_2 = Entry(frame_1, width=8, justify='right')
    entry_2.grid(row=1, column=1)

    entry_3 = Entry(frame_1, width=18, justify='right')
    entry_3.grid(row=1, column=2)

    entry_4 = Entry(frame_1, width=15, justify='right')
    entry_4.grid(row=1, column=3)

    win.mainloop()


if __name__ == '__main__':
    main()

