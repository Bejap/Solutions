from sqlalchemy.orm import Session
from sqlalchemy import select, extract
import plusbus_data as pbd
import plusbus_sql as pbsql

def booked_travels(travels, date_):
    with Session() as session:
        records = session.scalars(
            select(pbd.Travels).where(pbd.Travels.id == travels.id))
        booked_seats = 0
        for record in records:
            booked_seats += pbsql.get_record(pbd.Bookings, record.booking_id).booked_seats
    return booked_seats


def capacity_available(bookings, travel_id, new_booking):
    max_seats = travel_id.capacity
    booked = booked_travels(bookings)
    return max_seats >= booked + new_booking
