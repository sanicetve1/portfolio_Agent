
from sqlalchemy import Column, Integer, String, Float, Text, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True)
    input_hash = Column(String, unique=True, index=True)
    risk_rating = Column(String)
    sectors = Column(String)
    esg_preference = Column(String)
    portfolio = Column(JSON)
    summary = Column(Text)
    suggestions = Column(JSON)

# DB setup
engine = create_engine("sqlite:///recommendations.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
