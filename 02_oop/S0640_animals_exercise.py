import random
# Opgave 1
class Animal:
    def __init__(self, name, sound, height, weight, legs, female):
        self.name = name
        self.sound = sound
        self.height = height
        self.weight = weight
        self.legs = legs
        self.female = female

# Opgave 2
    def __repr__(self):
        return f'''
Name: {self.name}\n
Sound: {self.sound}\n
Height: {self.height}\n
Weight: {self.weight}\n
Legs: {self.legs}\n
Female: {self.female}
                                '''

# Opgave 3
    def make_noise(self):
        print(self.sound)

# Opgave 4
class Dog(Animal):
    def __init__(self, name, sound, height, weight, legs, female, tail_length, hunts_sheep):
        super().__init__(name, sound, height, weight, legs, female)
        self.tail_length = tail_length
        self.hunts_sheep = hunts_sheep

# Opgave 5
    def __repr__(self):
        animal_info = super().__repr__()
        return animal_info + f'''
Tail Length: {self.tail_length}\n
Hunts sheep: {self.hunts_sheep}
        '''
# Opgave 6

    def make_noise(self):
        print(self.sound)

# Opgave 7
    def wag_tail(self):
        print("The dog {0} is waging its {1} cm long tail!".format(self.name, self.tail_length))

# Opgave 9

    def __add__(self, other):
        if isinstance(other, Dog):
            child = "something"
            something = Mate(child, self, other)
            return something
        else:
            return "Doesn't work like that."

# Opgave 8

def _hunter(f, m):
    the_factor = random.random()
    if f:
        if m:
            return the_factor < 0.95
        else:
            return the_factor < 0.65
    else:
        if m:
            return the_factor < 0.4
        else:
            return the_factor < 0.05


class Mate(Dog):
    def __init__(self, name, f, m):
        if isinstance(f, Dog) and f.female != m.female:
            fh, fw, ft, fhs = f.height, f.weight, f.tail_length, f.hunts_sheep
            mh, mw, mt, mhs = m.height, m.weight, m.tail_length, m.hunts_sheep

            if random.random() < 0.5:
                gender = True
            else:
                gender = False

            ch = (fh + mh) / 2 + random.random() / 4 if gender else (fh + mh) / 2 - random.random() / 4
            cw = (fw + mw) / 2 + random.random() / 4 if gender else (fw + mw) / 2 - random.random() / 4
            ct = (ft + mt) / 2 + random.random() / 4 if gender else (ft + mt) / 2 - random.random() / 4
            chs = _hunter(fhs, mhs)
            super().__init__(name, "vov", ch, cw, 4, gender, ct, chs)
        else:
            print("The dogs cannot mate because they are of the same gender.")


# Opgave 10
def mating(name, f, m):

    child = Mate(name, f, m)
    return child

def main():
    mother = Dog("hannah", "vov", 1.2, 32, 4, True, 15, True)
    father = Dog("Carl", "vov", 0.7, 45, 4, False, 23, True)

    cho = mating("charlie", mother, father)

    fido = mother + father
    print(fido)
    print(cho)


main()
