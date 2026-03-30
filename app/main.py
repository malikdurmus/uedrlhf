import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"


DATABASE_URL = f"sqlite:///{DATA_DIR / 'feedback.db'}"
# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- DATABASE MODEL ---
class PreferencePair(Base):
    __tablename__ = "preference_pairs"
    pair_id = Column(Integer, primary_key=True, autoincrement=True)
    env_1_id = Column(String, nullable=False)
    env_2_id = Column(String, nullable=False)
    split = Column(String, nullable=False)
    status = Column(String, default="pending")
    
    # Feedback data
    preference = Column(Float, nullable=True) # 1.0 (Left/Env1), 0.0 (Right/Env2), 0.5 (Tie)
    x_pct = Column(Float, nullable=True)      # x position (for xai later on)
    y_pct = Column(Float, nullable=True)      # y position (for xai later on)
    time_ms = Column(Float, nullable=True)    # time taken for feedback selection
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# --- PYDANTIC VALIDATION ---
class FeedbackInput(BaseModel):
    pair_id: int  # We now track the pair, not just a single image ID
    preference: float = Field(..., ge=0, le=1) 
    x_pct: float = Field(..., ge=0, le=100)
    y_pct: float = Field(..., ge=0, le=100)
    time_ms: float = Field(..., ge=0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create tables if they don't exist
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
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# --- ROUTES ---
@app.get("/")
async def index():
    return FileResponse(Path(__file__).resolve().parent / "index.html")

@app.get("/next-pair")
async def next_pair():
    db = SessionLocal()
    try:
        # Fetch the first pair that hasn't been completed yet
        pair = db.query(PreferencePair).filter(PreferencePair.status == "pending").first()
        
        if not pair:
            return {"error": "No more pending pairs available! Training set complete."}
        
        return {
            "pair_id": pair.pair_id,
            "left": f"{pair.env_1_id}.png",
            "right": f"{pair.env_2_id}.png"
        }
    finally:
        db.close()


@app.post("/submit")
async def submit(data: FeedbackInput):
    db = SessionLocal()
    try:
        # Find the specific pair we are updating
        pair = db.query(PreferencePair).filter(PreferencePair.pair_id == data.pair_id).first()
        
        if not pair:
            return {"error": "Pair not found in database."}
        
        # Update the row with human feedback
        pair.preference = data.preference
        pair.x_pct = data.x_pct
        pair.y_pct = data.y_pct
        pair.time_ms = data.time_ms
        pair.status = "completed"  # Mark as done so it isn't served again
        pair.timestamp = datetime.now(timezone.utc)
        
        db.commit()
    finally:
        db.close()
        
    return {"status": "ok"}
