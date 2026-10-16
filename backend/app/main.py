"""
FastAPI Main Application Entry Point.
Production serving and streaming endpoints.
Checkpoint: Dynamic epoch calibration active.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.app.api.router import router as api_router
from backend.app.api.websocket_live import live_translation_websocket
from backend.app.core.config import settings
from backend.app.services.inference_service import InferenceService

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Professional Indian Sign Language Recognition & Translation Engine to English and Telugu.",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Live WebSocket Endpoint
app.add_api_websocket_route("/api/v1/live/ws", live_translation_websocket)


@app.on_event("startup")
async def startup_event():
    print(f"[{settings.PROJECT_NAME}] Starting up backend on {settings.HOST}:{settings.PORT}")
    # Warm up inference service singleton
    try:
        InferenceService.get_instance()
    except Exception as e:
        print(f"Warning: Model warm-up encountered: {e}")


@app.get("/")
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
