"""Opgave "Lunar arithmetic"

Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Denne øvelse er en valgfri udfordring for de fremragende programmører blandt jer.
Du behøver absolut ikke at løse denne øvelse for at fortsætte med succes.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Del 1:
    Se de første 3 minutter af denne video:
    https://www.youtube.com/watch?v=cZkGeR9CWbk

Del 2:
    Skriv en klasse Lunar_int(), med metoder, der gør, at du kan anvende operatorerne + og * på
    objekter af denne klasse, og at resultaterne svarer til de regler, der forklares i videoen.

Del 3:
    Se resten af videoen.

Del 4:
    Skriv en funktion calc_lunar_primes(n), som retunerer en liste med de første n lunar primes.


Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-meddelelse til din lærer: <filename> færdig
Fortsæt derefter med den næste fil."""

class Lunar_int:
    def __init__(self, t):
        self.t_list = [int(i) for i in str(t)]
        self.list_o_numb = self.t_list.copy()  # Directly assign the list instead of wrapping it in another list

    def suit(self, max_l):
        while len(self.list_o_numb) < max_l:
            self.list_o_numb.insert(0, 0)

    def add(self, other):
        result = []
        carry = 0

        max_length = max(len(self.list_o_numb), len(other.list_o_numb))
        self.suit(max_length)
        other.suit(max_length)

        for digit1, digit2 in zip(self.list_o_numb[::-1], other.list_o_numb[::-1]):
            total = digit1 + digit2 + carry
            carry = total // 10
            result.insert(0, total % 10)

        # Add carry if any
        if carry:
            result.insert(0, carry)

        return result

    def __repr__(self):
        return str(self.list_o_numb)


def create_lunar_list(numbers):
    lunar_list = []
    for num in numbers:
        lunar_list.append(Lunar_int(num))
    return lunar_list


def add_lunar_list(lunar_list):
    result = 0  # Initialize result as a number, not a list
    for lunar_int in lunar_list:
        result = lunar_int.add(Lunar_int(0))  # Pass result as a number
    return result



# Example usage:
numbers = [871, 654, 102]
lunar_list = create_lunar_list(numbers)
print("Lunar List:", lunar_list)

result = add_lunar_list(lunar_list)
print("Sum of Lunar List:", result)
