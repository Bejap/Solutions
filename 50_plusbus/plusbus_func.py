from sqlalchemy.orm import Session
from sqlalchemy import select, extract
import plusbus_data as pbd
import plusbus_sql as pbsql

def booked_travels(travel):
    travel_id = travel.id
    with Session(pbsql.engine) as session:
        records = session.scalars(
            select(pbd.Bookings.booked_seats).where(pbd.Bookings.travel_id == travel_id)
        ).all()
        booked_seats = sum(records)
    return booked_seats


def capacity_available(travel, new_booking):
    booked = booked_travels(travel)
    new_booking = int(new_booking)
    return travel.capacity >= booked + new_booking
