def pyramid(pyra_st, amount):
    srow = pyra_st.split()  # splits the string into a list of numbers with type str
    row = [int(q) for q in srow]
    new_row = row
    for i in range(1, amount + 1):  # initiates a loop for the amount of rows
        u = 0  # creates an index shift
        for j in range(len(row)):
            if j < len(row) - 1 and row[j] + row[j + 1] == i:
                new_row.insert(u + j + 1, i)
                u += 1

        row = new_row.copy()
        print(f"row {i}: {row}")


user_p = input("Please write your pyramid separated by spaces: ")
user_a = int(input("Please insert the amount of row you want: "))
pyramid(user_p, user_a)
