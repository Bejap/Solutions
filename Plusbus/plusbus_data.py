from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from dateutil import parser
from tkinter import messagebox

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Integer, primary_key=True)
    last_name = Column(String)
    phone_number = Column(String)

    def __repr__(self):
        return f"Customer({self.id=},    {self.last_name=},      {self.phone_number=})"

    def convert_to_tuple(self):
        return self.id, self.last_name, self.phone_number

    def valid(self):
        try:
            value = int(self.phone_number)
        except ValueError:
            return False
        return value >= 0

    @staticmethod
    def convert_from_tuple(tuple_):
        customer = Customer(id=tuple_[0], last_name=tuple_[1], phone_number=tuple_[2])
        return customer


class Travels(Base):
    __tablename__ = 'travels'
    id = Column(Integer, primary_key=True)
    route = Column(String)
    date = Column(Date)
    capacity = Column(Integer)

    def __repr__(self):
        return f"Travels({self.route=}, {self.date=}, {self.capacity=})"

    def convert_to_tuple(self):
        return self.route, self.date, self.capacity

    def valid(self):
        try:
            value = int(self.capacity)
        except ValueError:
            return False
        return value >= 0

    @staticmethod
    def convert_from_tuple(tuple_):
        travels = Travels(route=tuple_[0], date=tuple_[1], capacity=tuple_[2])
        return travels

class Bookings(Base):
    __tablename__ = 'bookings'
    id = Column(Integer, primary_key=True)
    travel_id = Column(Integer, ForeignKey('travels.id'), nullable=False)
    booked_seats = Column(Integer)

    def __repr__(self):
        return f"Bookings({self.id=}, {self.travel_id=}, {self.booked_seats=}"

    def convert_to_tuple(self):
        return self.id, self.travel_id, self.booked_seats

    def valid(self):
        try:
            value = int(self.booked_seats)
        except ValueError:
            return False
        return value >= 0

    @staticmethod
    def convert_from_tuple(tuple_):
        bookings = Bookings(id=tuple_[0], travel_id=tuple_[1], booked_seats=tuple_[2])
        return bookings
