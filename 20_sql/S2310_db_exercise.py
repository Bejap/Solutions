"""
Som altid skal du læse hele opgavebeskrivelsen omhyggeligt, før du begynder at løse opgaven.

Kopier denne fil til din egen løsningsmappe. Skriv din løsning ind i kopien.

Anvend det, du har lært i dette kapitel om databaser, på en første opgave.

Trin 1:
Opret en ny SQLite database "S2311_my_second_sql_database.db" i din solutions mappe.
Denne database skal indeholde 2 tabeller.
Den første tabel skal hedde "customers" og repræsenteres i Python-koden af en klasse kaldet "Customer".
Tabellen bruger sin første attribut "id" som primærnøgle.
De andre attributter i tabellen hedder "name", "address" og "age".
Definer selv fornuftige datatyper for attributterne.

Trin 2:
Den anden tabel skal hedde "products" og repræsenteres i Python-koden af en klasse kaldet "Product".
Denne tabel bruger også sin første attribut "id" som primærnøgle.
De andre attributter i tabellen hedder "product_number", "price" og "brand".

Trin 3:
Skriv en funktion create_test_data(), der opretter testdata for begge tabeller.

Trin 4:
Skriv en metode __repr__() for begge dataklasser, så du kan vise poster til testformål med print().

Til læsning fra databasen kan du genbruge de to funktioner select_all() og get_record() fra S2240_db_class_methods.py.

Trin 5:
Skriv hovedprogrammet: Det skriver testdata til databasen, læser dataene fra databasen med select_all() og/eller get_record() og udskriver posterne til konsollen med print().

Når dit program er færdigt, skal du skubbe det til dit github-repository.
Send derefter denne Teams-besked til din lærer: <filename> færdig
Fortsæt derefter med den næste fil.
"""
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy import Column, String, Integer
from sqlalchemy import create_engine, select

database = 'sqlite:///S2311_my_second_sql_database.db'
Base = declarative_base()


class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    address = Column(String)
    age = Column(Integer)

    def __repr__(self):
        return f'Customer(ID: {self.id}, Name: {self.name}, Address: {self.address}, Age: {self.age})'


class Product(Base):
    __tablename__ = 'product'
    id = Column(Integer, primary_key=True)
    product_number = Column(Integer)
    price = Column(Integer)
    brand = Column(String)

    def __repr__(self):
        return f'Product(ID: {self.id}, ProductNumber: {self.product_number}, Price: {self.price}, Brand: {self.brand}'


def create_data():
    with Session(engine) as session:
        new_items = [
            Customer(name="peter", address="Somevile 18", age=18),
            Customer(name="susan", address="Nowhere 20", age=19),
            Customer(name="jane", address="Banegaarden 91", age=21),
            Customer(name="harry", address="Knowtowho 2", age=20)
        ]
        new_products = [
            Product(product_number=472831, price=5000, brand="Brandy"),
            Product(product_number=182930, price=2000, brand="Sweet"),
            Product(product_number=940231, price=7800, brand="KOTAK")
        ]
        session.add_all(new_items)
        session.add_all(new_products)
        session.commit()


def select_all(classparam):
    with Session(engine) as session:
        records = session.scalars(select(classparam))
        result = []
        for record in records:
            result.append(record)
    return result


def get_record(classparam, record_id):
    with Session(engine) as session:
        record = session.scalars(select(classparam).where(classparam.id == record_id)).first()
    return record


engine = create_engine(database, echo=False, future=True)
Base.metadata.create_all(engine)
create_data()


def printing():
    print(get_record(Customer, 3))
    print(get_record(Product, 1))
    print(select_all(Customer))
    print(select_all(Product))


printing()
