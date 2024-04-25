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
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.t_list = [int(i) for i in str(self.t)]
        self.y_list = [int(j) for j in str(self.y)]
        if len(self.y_list) > len(self.t_list):
            index = 0
            for _ in range(len(self.y_list) - len(self.t_list)):
                self.t_list.insert(index, 0)
                index += 1
        elif len(self.y_list) < len(self.t_list):
            index = 0
            for _ in range(len(self.t_list) - len(self.y_list)):
                self.y_list.insert(index, 0)
                index += 1
        else:
            pass

    def add(self):
        new_numb = []
        for k in range(len(self.t_list)):
            if self.y_list[k] < self.t_list[k]:
                new_numb.append(self.t_list[k])
            else:
                new_numb.append(self.y_list[k])

        return int(''.join(map(str, new_numb)))

    def multiply(self):
        new_numb = []
        for j in range(len(self.t_list)):  # creates a loop that iterates as the length of the first integer
            list_o_new = []
            for i in range(len(self.y_list)):  # creates a loop to iterate through all possibilities
                if self.t_list[j] < self.y_list[i]:
                    list_o_new.append(self.t_list[j])
                else:
                    list_o_new.append(self.y_list[i])
            new_numb.append(list_o_new)

        return new_numb



carl = Lunar_int(725, 129)
print(carl.add())
print(carl.multiply())
