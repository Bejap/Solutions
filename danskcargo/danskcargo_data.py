from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer

Base = declarative_base()

class Container(Base):
    __tablename__ = 'container'
    id = Column(Integer, primary_key=True)
    weight = Column(Integer)
    destination = Column(String)

    def convert_to_tuple(self):
        return self.id, self.weight, self.destination
