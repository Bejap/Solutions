
def file_reading(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    line_number = 0
    for line in lines:
        line_number += 1
        line_list = line.strip("\n").split(" ")
        print("{0} is {1} years old\n".format(line_list[0], line_list[1]))

file_reading("tekstfil.txt")