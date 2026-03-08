"""
DocEnhance AI — FastAPI application entry point
================================================
Registers each feature's APIRouter.
To add a new feature, create routers/<your_feature>.py and include it here.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import enhancement  # ← Image Enhancement (IT22348098)
# from routers import your_feature   # ← add your own router here

app = FastAPI(title="DocEnhance AI", version="1.0.0")

# CORS — allow React dev server (restrict to your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register feature routers ──────────────────────────────────────────────────
app.include_router(enhancement.router)
# app.include_router(your_feature.router)

