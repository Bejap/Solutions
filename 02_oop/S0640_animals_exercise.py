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

# Opgave 8
    def mate(self, father, mother):
        if type(father) == type(mother):
            f_tail = father._getTail()
            m_tail = mother._getTail()
            c_tail = (f_tail + m_tail)/2 *( (random.randrange(6, 12)/10))





# Opgave 9
# Opgave 10




def main():
    M_dog = Dog( "Carl", "vov", 2, 45, 4, True, 15, True)
    print(my_dog)
    my_dog.wag_tail()




main()
