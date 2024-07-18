from sqlalchemy.future import engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select, update, delete
from plusbus_data import Customer, Travels, Bookings, Base
from sqlalchemy.engine import Engine
from datetime import date
from sqlalchemy import event


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


Database = 'sqlite:///plusbus.db'


def create_test_data():
    with Session(engine) as session:
        new_item = []
        new_item.append(Customer(last_name="Smith", phone_number="98765432"))
        new_item.append(Customer(last_name="Larsson", phone_number="12345678"))
        a_date = date(day=26, month=7, year=2023)
        new_item.append(Travels(route="Aab-Aar", date=a_date, capacity=48))
        new_item.append(Bookings(travel_id=1, booked_seats=5))
        session.add_all(new_item)
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


def create_record(record):
    with Session(engine) as session:
        record.id = None
        session.add(record)
        session.commit()



# region customer
def update_customer(customer):
    with Session(engine) as session:
        session.execute(update(Customer).where(Customer.id == customer.id).values(last_name=customer.last_name, phone_number=customer.phone_number))
        session.commit()


def delete_customer(customer):
    with Session(engine) as session:
        session.execute(delete(Customer).where(Customer.id == customer.id))
        session.commit()


# endregion

# region travels
def update_travels(travels):
    with Session(engine) as session:
        session.execute(update(Travels).where(Travels.id == travels.id).values(route=travels.route, date=travels.date, capacity=travels.capacity))
        session.commit()


def delete_travels(travels):
    with Session(engine) as session:
        session.execute(delete(Travels).where(Travels.id == travels.id))
        session.commit()


# endregion

# region bookings
def update_bookings(bookings):
    with Session(engine) as session:
        session.execute(update(Bookings).where(Bookings.id == bookings.id).values(travel_id=bookings.travel_id, booked_seats=bookings.booked_seats))
        session.commit()


def delete_bookings(bookings):
    with Session(engine) as session:
        session.delete(delete(Bookings).where(Bookings.id == bookings.id))
        session.commit()


# endregion

if __name__ == '__main__':
    engine = create_engine(Database, echo=False, future=True)
    Base.metadata.create_all(engine)
    create_test_data()
    print(select_all(Customer))
    print(get_record(Customer, 2))
else:
    engine = create_engine(Database, echo=False, future=True)
    Base.metadata.create_all(engine)
