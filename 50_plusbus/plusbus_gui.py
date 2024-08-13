import tkinter as tk
from tkinter import ttk, messagebox
import plusbus_data as pbd
import plusbus_sql as pbsql
import plusbus_func as pbf

# region global constants
padx = 8
pady = 4
rowheight = 24
treeview_background = "#eeeeee"
treeview_foreground = "black"
treeview_selected = "206030"
oddrow = "#dddddd"
evenrow = "#cccccc"
# endregion

# region common widgets
main_window = tk.Tk()
main_window.title("AspIT S2: 50_plusbus")
main_window.geometry("1350x500")

style = ttk.Style()
style.theme_use("default")

style.configure("Treeview", background=treeview_background, foreground=treeview_foreground, rowheight=rowheight,
                fieldbackground=treeview_background)
style.map("Treeview", background=[('selected', treeview_selected)])


# endregion

# region common functions
def read_table(tree, class_):
    count = 0
    result = pbsql.select_all(class_)
    for record in result:
        if record.valid():
            if count % 2 == 0:
                tree.insert(parent='', index='end', iid=str(count), text='', values=record.convert_to_tuple(), tags=('evenrow',))
            else:
                tree.insert(parent='', index='end', iid=str(count), text='', values=record.convert_to_tuple(), tags=('oddrow',))
            count += 1


def empty_treeview(tree):
    tree.delete(*tree.get_children())


def refresh_treeview(tree, class_):
    empty_treeview(tree)
    read_table(tree, class_)


# endregion

# region customer fuctions
def read_customer_entries():
    return entry_customer_id.get(), entry_customer_last_name.get(), entry_customer_phone_number.get()


def clear_customer_entries():
    entry_customer_id.delete(0, tk.END)
    entry_customer_last_name.delete(0, tk.END)
    entry_customer_phone_number.delete(0, tk.END)


def write_customer_entries(values):
    entry_customer_id.insert(0, values[0])
    entry_customer_last_name.insert(0, values[1])
    entry_customer_phone_number.insert(0, values[2])


def edit_customer(event, tree):
    index_selected = tree.focus()
    values = tree.item(index_selected, 'values')

    clear_customer_entries()
    write_customer_entries(values)


def create_customer(tree, record):
    customer = pbd.Customer.convert_from_tuple(record)
    pbsql.create_record(customer)
    clear_customer_entries()
    refresh_treeview(tree, pbd.Customer)


def update_customer(tree, record):
    customer = pbd.Customer.convert_from_tuple(record)
    pbsql.update_customer(customer)
    clear_customer_entries()
    refresh_treeview(tree, pbd.Customer)


def delete_customer(tree, record):
    customer = pbd.Customer.convert_from_tuple(record)
    pbsql.delete_customer(customer)
    clear_customer_entries()
    refresh_treeview(tree, pbd.Customer)


# endregion

# region travels fuctions
def read_travel_entries():
    return entry_travel_id.get(), entry_travel_route.get(), entry_travel_date.get(), entry_travel_capacity.get()


def clear_travel_entries():
    entry_travel_id.delete(0, tk.END)
    entry_travel_route.delete(0, tk.END)
    entry_travel_date.delete(0, tk.END)
    entry_travel_capacity.delete(0, tk.END)


def write_travel_entries(values):
    entry_travel_id.insert(0, values[0])
    entry_travel_route.insert(0, values[1])
    entry_travel_date.insert(0, values[2])
    entry_travel_capacity.insert(0, values[3])


def edit_travel(event, tree):
    index_selected = tree.focus()
    values = tree.item(index_selected, 'values')

    clear_travel_entries()
    write_travel_entries(values)


def create_travel(tree, record):
    travel = pbd.Travels.convert_from_tuple(record)
    pbsql.create_record(travel)
    clear_travel_entries()
    refresh_treeview(tree, pbd.Travels)


def update_travel(tree, record):
    travel = pbd.Travels.convert_from_tuple(record)
    pbsql.update_travels(travel)
    clear_travel_entries()
    refresh_treeview(tree, pbd.Travels)


def delete_travel(tree, record):
    travel = pbd.Travels.convert_from_tuple(record)
    pbsql.delete_travels(travel)
    clear_travel_entries()
    refresh_treeview(tree, pbd.Travels)


# endregion

# region bookings fuctions
def read_booking_entries():
    return entry_booking_id.get(), entry_booking_travel_id.get(), entry_booking_customer_id.get(), entry_booking_booked_seats.get()


def clear_booking_entries():
    entry_booking_id.delete(0, tk.END)
    entry_booking_travel_id.delete(0, tk.END)
    entry_booking_customer_id.delete(0, tk.END)
    entry_booking_booked_seats.delete(0, tk.END)


def write_booking_entries(values):
    entry_booking_id.insert(0, values[0])
    entry_booking_travel_id.insert(0, values[1])
    entry_booking_customer_id.insert(0, values[2])
    entry_booking_booked_seats.insert(0, values[3])


def edit_booking(event, tree):
    index_selected = tree.focus()
    values = tree.item(index_selected, 'values')

    clear_booking_entries()
    write_booking_entries(values)


def create_booking(tree, record):
    booking = pbd.Bookings.convert_from_tuple(record)

    travel = pbsql.get_record(pbd.Travels, booking.travel_id)
    customer = pbsql.get_record(pbd.Customer, booking.customer_id)

    if travel is None:
        messagebox.showinfo('Error', 'Travel does not exist')
        return

    if customer is None:
        messagebox.showinfo('Error', 'Customer does not exist')
        return

    if not pbf.capacity_available(travel, booking.booked_seats):
        messagebox.showinfo('Error', 'Not enough capacity available')
        return

    pbsql.create_record(booking)
    clear_booking_entries()
    refresh_treeview(tree, pbd.Bookings)


def update_booking(tree, record):
    booking = pbd.Bookings.convert_from_tuple(record)

    travel = pbsql.get_record(pbd.Travels, booking.travel_id)
    customer = pbsql.get_record(pbd.Customer, booking.customer_id)

    if travel is None:
        messagebox.showinfo('Error', 'Travel does not exist')
        return

    if customer is None:
        messagebox.showinfo('Error', 'Customer does not exist')
        return

    if not pbf.capacity_available(travel, booking.booked_seats):
        messagebox.showinfo('Error', 'Not enough capacity available')
        return

    pbsql.update_bookings(booking)
    clear_booking_entries()
    refresh_treeview(tree, pbd.Bookings)


def delete_booking(tree, record):
    booking = pbd.Bookings.convert_from_tuple(record)
    pbsql.delete_bookings(booking)
    clear_booking_entries()
    refresh_treeview(tree, pbd.Bookings)


# endregion


# region customer widgets
frame_customer = tk.LabelFrame(main_window, text="Customer")
frame_customer.grid(row=0, column=0, padx=padx, pady=pady, sticky=tk.N)

tree_frame_customer = tk.Frame(frame_customer)
tree_frame_customer.grid(row=0, column=0, padx=padx, pady=pady)
tree_scroll_customer = tk.Scrollbar(tree_frame_customer)
tree_scroll_customer.grid(row=0, column=1, padx=0, pady=pady, sticky='ns')
tree_customer = ttk.Treeview(tree_frame_customer, yscrollcommand=tree_scroll_customer.set, selectmode='browse')
tree_customer.grid(row=0, column=0, padx=0, pady=pady)
tree_scroll_customer.config(command=tree_customer.yview)

tree_customer['columns'] = ("id", "last name", "phone number")
tree_customer.column("#0", width=0, stretch=tk.NO)
tree_customer.column("id", anchor=tk.E, width=30)
tree_customer.column("last name", anchor=tk.E, width=100)
tree_customer.column("phone number", anchor=tk.W, width=180)
tree_customer.heading("#0", text="", anchor=tk.W)
tree_customer.heading("id", text="ID", anchor=tk.CENTER)
tree_customer.heading("last name", text="Last name", anchor=tk.CENTER)
tree_customer.heading("phone number", text="Phone number", anchor=tk.CENTER)
tree_customer.tag_configure('oddrow', background=oddrow)
tree_customer.tag_configure('evenrow', background=evenrow)
tree_customer.bind("<ButtonRelease-1>", lambda event: edit_customer(event, tree_customer))

controls_frame_customer = tk.Frame(frame_customer)
controls_frame_customer.grid(row=3, column=0, padx=padx, pady=pady)

edit_frame_customer = tk.Frame(controls_frame_customer)
edit_frame_customer.grid(row=0, column=0, padx=padx, pady=pady)

label_customer_id = tk.Label(edit_frame_customer, text="Id")
label_customer_id.grid(row=0, column=0, padx=padx, pady=pady)
entry_customer_id = tk.Entry(edit_frame_customer, width=4, justify="right")
entry_customer_id.grid(row=1, column=0, padx=padx, pady=pady)

label_customer_last_name = tk.Label(edit_frame_customer, text="Last name")
label_customer_last_name.grid(row=0, column=1, padx=padx, pady=pady)
entry_customer_last_name = tk.Entry(edit_frame_customer, width=12, justify="right")
entry_customer_last_name.grid(row=1, column=1, padx=padx, pady=pady)

label_customer_phone_number = tk.Label(edit_frame_customer, text="Phone number")
label_customer_phone_number.grid(row=0, column=2, padx=padx, pady=pady)
entry_customer_phone_number = tk.Entry(edit_frame_customer, width=24, justify="right")
entry_customer_phone_number.grid(row=1, column=2, padx=padx, pady=pady)

button_frame_customer = tk.Frame(controls_frame_customer)
button_frame_customer.grid(row=1, column=0, padx=padx, pady=pady)

button_create_customer = tk.Button(button_frame_customer, text="Create", command=lambda: create_customer(tree_customer, read_customer_entries()))
button_create_customer.grid(row=0, column=1, padx=padx, pady=pady)

button_update_customer = tk.Button(button_frame_customer, text="Update", command=lambda: update_customer(tree_customer, read_customer_entries()))
button_update_customer.grid(row=0, column=2, padx=padx, pady=pady)

button_delete_customer = tk.Button(button_frame_customer, text="Delete", command=lambda: delete_customer(tree_customer, read_customer_entries()))
button_delete_customer.grid(row=0, column=3, padx=padx, pady=pady)

button_clear_boxes = tk.Button(button_frame_customer, text="Clear Entry Boxes", command=clear_customer_entries)
button_clear_boxes.grid(row=0, column=4, padx=padx, pady=pady)
# endregion

# region travel widgets
frame_travel = tk.LabelFrame(main_window, text="Travel")
frame_travel.grid(row=0, column=1, padx=padx, pady=pady, sticky=tk.N)

tree_frame_travel = tk.Frame(frame_travel)
tree_frame_travel.grid(row=0, column=0, padx=padx, pady=pady)
tree_scroll_travel = tk.Scrollbar(tree_frame_travel)
tree_scroll_travel.grid(row=0, column=1, padx=0, pady=pady, sticky='ns')
tree_travel = ttk.Treeview(tree_frame_travel, yscrollcommand=tree_scroll_travel.set, selectmode='browse')
tree_travel.grid(row=0, column=0, padx=0, pady=pady)
tree_scroll_travel.config(command=tree_travel.yview)

tree_travel['columns'] = ("id", "route", "date", "capacity")
tree_travel.column("#0", width=0, stretch=tk.NO)
tree_travel.column("id", anchor=tk.E, width=30)
tree_travel.column("route", anchor=tk.E, width=150)
tree_travel.column("date", anchor=tk.W, width=100)
tree_travel.column("capacity", anchor=tk.W, width=130)
tree_travel.heading("#0", text="", anchor=tk.W)
tree_travel.heading("id", text="ID", anchor=tk.CENTER)
tree_travel.heading("route", text="Route", anchor=tk.CENTER)
tree_travel.heading("date", text="Date", anchor=tk.CENTER)
tree_travel.heading("capacity", text="Capacity", anchor=tk.CENTER)
tree_travel.tag_configure('oddrow', background=oddrow)
tree_travel.tag_configure('evenrow', background=evenrow)
tree_travel.bind("<ButtonRelease-1>", lambda event: edit_travel(event, tree_travel))

controls_frame_travel = tk.Frame(frame_travel)
controls_frame_travel.grid(row=3, column=0, padx=padx, pady=pady)

edit_frame_travel = tk.Frame(controls_frame_travel)
edit_frame_travel.grid(row=0, column=0, padx=padx, pady=pady)

label_travel_id = tk.Label(edit_frame_travel, text="Id")
label_travel_id.grid(row=0, column=0, padx=padx, pady=pady)
entry_travel_id = tk.Entry(edit_frame_travel, width=4, justify="right")
entry_travel_id.grid(row=1, column=0, padx=padx, pady=pady)

label_travel_route = tk.Label(edit_frame_travel, text="Route")
label_travel_route.grid(row=0, column=1, padx=padx, pady=pady)
entry_travel_route = tk.Entry(edit_frame_travel, width=12, justify="right")
entry_travel_route.grid(row=1, column=1, padx=padx, pady=pady)

label_travel_date = tk.Label(edit_frame_travel, text="Date")
label_travel_date.grid(row=0, column=2, padx=padx, pady=pady)
entry_travel_date = tk.Entry(edit_frame_travel, width=24, justify="right")
entry_travel_date.grid(row=1, column=2, padx=padx, pady=pady)

label_travel_capacity = tk.Label(edit_frame_travel, text="Capacity")
label_travel_capacity.grid(row=0, column=3, padx=padx, pady=pady)
entry_travel_capacity = tk.Entry(edit_frame_travel, width=24, justify="right")
entry_travel_capacity.grid(row=1, column=3, padx=padx, pady=pady)

button_frame_travel = tk.Frame(controls_frame_travel)
button_frame_travel.grid(row=1, column=0, padx=padx, pady=pady)

button_create_travel = tk.Button(button_frame_travel, text="Create", command=lambda: create_travel(tree_travel, read_travel_entries()))
button_create_travel.grid(row=0, column=1, padx=padx, pady=pady)

button_update_travel = tk.Button(button_frame_travel, text="Update", command=lambda: update_travel(tree_travel, read_travel_entries()))
button_update_travel.grid(row=0, column=2, padx=padx, pady=pady)

button_delete_travel = tk.Button(button_frame_travel, text="Delete", command=lambda: delete_travel(tree_travel, read_travel_entries()))
button_delete_travel.grid(row=0, column=3, padx=padx, pady=pady)

button_clear_boxes = tk.Button(button_frame_travel, text="Clear Entry Boxes", command=clear_travel_entries)
button_clear_boxes.grid(row=0, column=4, padx=padx, pady=pady)
# endregion

# region booking widgets
frame_booking = tk.LabelFrame(main_window, text="Booking")
frame_booking.grid(row=0, column=2, padx=padx, pady=pady, sticky=tk.N)

tree_frame_booking = tk.Frame(frame_booking)
tree_frame_booking.grid(row=0, column=0, padx=padx, pady=pady)
tree_scroll_booking = tk.Scrollbar(tree_frame_booking)
tree_scroll_booking.grid(row=0, column=1, padx=0, pady=pady, sticky='ns')
tree_booking = ttk.Treeview(tree_frame_booking, yscrollcommand=tree_scroll_booking.set, selectmode='browse')
tree_booking.grid(row=0, column=0, padx=0, pady=pady)
tree_scroll_booking.config(command=tree_booking.yview)

tree_booking['columns'] = ("id", "travel id", "customer id", "booked seats")
tree_booking.column("#0", width=0, stretch=tk.NO)
tree_booking.column("id", anchor=tk.E, width=30)
tree_booking.column("travel id", anchor=tk.E, width=80)
tree_booking.column("booked seats", anchor=tk.E, width=120)
tree_booking.column("customer id", anchor=tk.E, width=80)
tree_booking.heading("#0", text="", anchor=tk.W)
tree_booking.heading("id", text="ID", anchor=tk.CENTER)
tree_booking.heading("travel id", text="Travel id", anchor=tk.CENTER)
tree_booking.heading("customer id", text="Customer id", anchor=tk.CENTER)
tree_booking.heading("booked seats", text="Booked seats", anchor=tk.CENTER)
tree_booking.tag_configure('oddrow', background=oddrow)
tree_booking.tag_configure('evenrow', background=evenrow)
tree_booking.bind("<ButtonRelease-1>", lambda event: edit_booking(event, tree_booking))

controls_frame_booking = tk.Frame(frame_booking)
controls_frame_booking.grid(row=3, column=0, padx=padx, pady=pady)

edit_frame_booking = tk.Frame(controls_frame_booking)
edit_frame_booking.grid(row=0, column=0, padx=padx, pady=pady)

label_booking_id = tk.Label(edit_frame_booking, text="ID")
label_booking_id.grid(row=0, column=0, padx=padx, pady=pady)
entry_booking_id = tk.Entry(edit_frame_booking, width=8)
entry_booking_id.grid(row=1, column=0, padx=padx, pady=pady)

label_booking_travel_id = tk.Label(edit_frame_booking, text="travel id")
label_booking_travel_id.grid(row=0, column=1, padx=padx, pady=pady)
entry_booking_travel_id = tk.Entry(edit_frame_booking, width=12, justify="right")
entry_booking_travel_id.grid(row=1, column=1, padx=padx, pady=pady)

label_booking_customer_id = tk.Label(edit_frame_booking, text="customer id")
label_booking_customer_id.grid(row=0, column=2, padx=padx, pady=pady)
entry_booking_customer_id = tk.Entry(edit_frame_booking, width=24, justify="right")
entry_booking_customer_id.grid(row=1, column=2, padx=padx, pady=pady)

label_booking_booked_seats = tk.Label(edit_frame_booking, text="Booked seats")
label_booking_booked_seats.grid(row=0, column=3, padx=padx, pady=pady)
entry_booking_booked_seats = tk.Entry(edit_frame_booking, width=12, justify="right")
entry_booking_booked_seats.grid(row=1, column=3, padx=padx, pady=pady)

button_frame_booking = tk.Frame(controls_frame_booking)
button_frame_booking.grid(row=1, column=0, padx=padx, pady=pady)

button_create_booking = tk.Button(button_frame_booking, text="Create", command=lambda: create_booking(tree_booking, read_booking_entries()))
button_create_booking.grid(row=0, column=1, padx=padx, pady=pady)

button_update_booking = tk.Button(button_frame_booking, text="Update", command=lambda: update_booking(tree_booking, read_booking_entries()))
button_update_booking.grid(row=0, column=2, padx=padx, pady=pady)

button_delete_booking = tk.Button(button_frame_booking, text="Delete", command=lambda: delete_booking(tree_booking, read_booking_entries()))
button_delete_booking.grid(row=0, column=3, padx=padx, pady=pady)

button_clear_boxes = tk.Button(button_frame_booking, text="Clear Entry Boxes", command=clear_booking_entries)
button_clear_boxes.grid(row=0, column=4, padx=padx, pady=pady)

# endregion

if __name__ == '__main__':
    refresh_treeview(tree_customer, pbd.Customer)
    refresh_treeview(tree_travel, pbd.Travels)
    refresh_treeview(tree_booking, pbd.Bookings)
    main_window.mainloop()
