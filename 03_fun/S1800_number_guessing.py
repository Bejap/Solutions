""" Opgave "Number guessing"

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Opret et program, der spiller et gættespil med brugeren. Programmet fungerer på følgende måde:
    Forklar reglerne for brugeren.
    Generer tilfældigt et 4-cifret heltal.
    Bed brugeren om at gætte et 4-cifret tal.
    Hvert ciffer, som brugeren gætter korrekt i den rigtige position, tæller som en sort mønt.
    Hvert ciffer, som brugeren gætter korrekt, men i den forkerte position, tæller som en hvid mønt.
    Når brugeren har gættet, udskrives det, hvor mange sorte og hvide mønter gættet er værd.
    Lad brugeren gætte, indtil gættet er korrekt.
    Hold styr på antallet af gæt, som brugeren gætter i løbet af spillet, og print det ud til sidst.

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""
import random
def get_number():
    return random.randint(1000, 9999)

def guess():
    bid = int(input("Guess a number between, 1000 and 9999: "))
    if bid < 1000:
        return bid


def evaluate_guess(n, g):
    p = [0, 0]
    g_str = str(g)
    n_str = str(n)

    # Count black coins (correct digit in correct position)
    for i in range(4):
        if g_str[i] == n_str[i]:
            p[0] += 1

    # Count white coins (correct digit in wrong position)
    for digit in set(g_str):  # Iterate over unique digits in guessed number
        if digit in n_str:  # Check if the digit is in the target number
            # Count occurrences of digit in guessed and target numbers
            guessed_count = g_str.count(digit)
            target_count = n_str.count(digit)
            # Count only if the digit is not in the correct position
            if g_str.index(digit) != n_str.index(digit):
                p[1] += min(guessed_count, target_count)

    # Adjust white coins for correct digits in correct position
    p[1] -= p[0]

    # Ensure that black coins count is non-negative
    p[0] = max(0, p[0])

    print(p)
    return p


def main():
    number = get_number()
    guesses = 0
    guessing = guess()
    kron = evaluate_guess(number, guessing)
    print("white coins", kron[0], "\nblack coins", kron[1])
    while guessing != number:
        guesses += 1
        guessing = guess()
        kron = evaluate_guess(number, guessing)
        print("white coins", kron[0], "\nblack coins", kron[1])
        if kron[1] == 4:
            print("you won, the number was indeed: ", number)
            break


main()
