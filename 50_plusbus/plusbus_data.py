from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from dateutil import parser
from tkinter import messagebox

Base = declarative_base() # Creating the registry and base classes


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
        return f"Travels({self.id=}, {self.route=}, {self.date=}, {self.capacity=})"

    def convert_to_tuple(self):
        return self.id, self.route, self.date, self.capacity

    def valid(self):
        try:
            value = int(self.capacity)
        except ValueError:
            return False
        return value >= 0

    @staticmethod
    def convert_from_tuple(tuple_):
        try:
            if tuple_[0] != "":
                id_ = int(tuple_[0])
            else:
                id_ = 0
            route = str(tuple_[1])
            date = parser.parse(tuple_[2])
            capacity = int(tuple_[3])
            travel = Travels(id=id_, route=route, date=date, capacity=capacity)
            return travel
        except:
            messagebox.showinfo("Error", "Please enter a valid date")


class Bookings(Base):
    __tablename__ = 'bookings'
    id = Column(Integer, primary_key=True)
    travel_id = Column(Integer, ForeignKey('travels.id'), nullable=False)
    customer_id = Column(Integer, ForeignKey('customer.id'), nullable=False)
    booked_seats = Column(Integer)

    def __repr__(self):
        return f"Bookings({self.id=}, {self.travel_id=}, {self.customer_id=} {self.booked_seats=}"

    def convert_to_tuple(self):
        return self.id, self.travel_id, self.customer_id, self.booked_seats

    def valid(self):
        try:
            value = int(self.booked_seats)
        except ValueError:
            return False
        return value >= 0

    @staticmethod
    def convert_from_tuple(tuple_):
        return Bookings(id=tuple_[0], travel_id=tuple_[1], customer_id=tuple_[2], booked_seats=tuple_[3])
