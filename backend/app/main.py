"""
DocEnhance AI — FastAPI application entry point
================================================
Created by the team. Each teammate registers their own router below.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Teammate routers — add yours here ─────────────────────────────────────────
from routes import enhancement  # IT22348098 — Medical Report Enhancement
# from routes import your_feature  # ← teammate: import your router here

app = FastAPI(title="DocEnhance AI", version="1.0.0")

# CORS — allow React dev server (restrict to your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────────
app.include_router(enhancement.router)   # IT22348098 — Medical Report Enhancement
# app.include_router(your_feature.router)  # ← teammate: register your router here
