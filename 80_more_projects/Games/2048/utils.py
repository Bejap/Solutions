from colorama import Fore, init

init(autoreset=True)


def boards_differ(b1, b2):
    return any(b1[r][c] != b2[r][c] for r in range(len(b1[0])) for c in range(len(b1[1])))


def colored_tile(val):
    if val == 0:
        return ".".rjust(4)
    elif val == 2:
        return Fore.WHITE + str(val).rjust(4)
    elif val == 4:
        return Fore.CYAN + str(val).rjust(4)
    elif val == 8:
        return Fore.GREEN + str(val).rjust(4)
    elif val == 16:
        return Fore.YELLOW + str(val).rjust(4)
    elif val == 32:
        return Fore.MAGENTA + str(val).rjust(4)
    elif val == 64:
        return Fore.RED + str(val).rjust(4)
    else:
        return Fore.LIGHTRED_EX + str(val).rjust(4)


def merge_row(row):
    """
    :param row:
    :return: the merged row shifted to left
    """
    new_row = [x for x in row if x != 0]

    i = 0
    while i < len(new_row) - 1:
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            new_row[i + 1] = 0
            i += 2
        else:
            i += 1

    final = [x for x in new_row if x != 0]
    final += [0] * (len(row) - len(final))
    return final


def transpose(board):
    """
    Swapping columns to rows and rows to columns
    :param board:
    :return: Transposed board, columns to rows
    """
    k = [row.copy() for row in board]
    for i in range(len(board)):
        for j in range(len(board)):
            k[j][i] = board[i][j]

    return k


assert merge_row([2, 2, 0, 0]) == [4, 0, 0, 0]
assert merge_row([2, 2, 4, 4]) == [4, 8, 0, 0]
assert merge_row([2, 0, 2, 2]) == [4, 2, 0, 0]

assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

assert boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 5]]) == True
assert boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 4]]) == False
