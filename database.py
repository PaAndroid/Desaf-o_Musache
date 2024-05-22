from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = 'mssql+pyodbc://@DESKTOP-I5VGBH8\\SQLEXPRESS2016/Python?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class QA(Base):
    __tablename__ = "qa"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(255), index=True)
    answer = Column(String(255))
