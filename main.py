import os
import random
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Database setup
DATABASE_URL = "sqlite:///feedback.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String, nullable=False)
    choice = Column(Float, nullable=False) # To-do: will modify this for neutral feedback.
    x_pct = Column(Float, nullable=False)
    y_pct = Column(Float, nullable=False)
    time_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# check w/ pydantic
class FeedbackInput(BaseModel):
    image_id: str
    choice: float = Field(..., ge = 0 , le = 1)
    x_pct: float = Field(..., ge=0, le=100)
    y_pct: float = Field(..., ge=0, le=100)
    time_ms: float = Field(..., ge=0)



# create images dir and database tables 
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve images
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ROUTES
@app.get("/")
async def index():
    return FileResponse(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    )

@app.get("/next-pair")
async def next_pair():
    valid_ext = {".png"}
    files = [
        f
        for f in os.listdir(IMAGES_DIR)  #right now randomly selecting files
        if os.path.isfile(os.path.join(IMAGES_DIR, f)) # might change this later on
        and os.path.splitext(f)[1].lower() in valid_ext #  (using comparsions that are more comparable)
    ]
    if len(files) < 2:
        return {"error": "Need at least 2 images in the images/ folder."}
    left, right = random.sample(files, 2)
    return {"left": left, "right": right}

@app.post("/submit")
async def submit(data: FeedbackInput):
    db = SessionLocal()
    try:
        row = Feedback(
            image_id=data.image_id,
            choice=data.choice,
            x_pct=data.x_pct,
            y_pct=data.y_pct,
            time_ms=data.time_ms,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()
    return {"status": "ok"}

