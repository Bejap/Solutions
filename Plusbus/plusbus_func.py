from sqlalchemy.orm import Session
from sqlalchemy import select, extract
import plusbus_data as pbd
import plusbus_sql as pbsql

def booked_travels(travels, date_):
    with Session() as session:
        records = session.scalars(
            select(pbd.Travels)
            .where(pbd.Travels.id == travels.id)
            .where(extract('day', pbd.Travels.date) == date_.day)
            .where(extract('month', pbd.Travels.date) == date_.month)
            .where(extract('year', pbd.Travels.date) == date_.year))
