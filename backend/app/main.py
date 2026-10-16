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


# Frontend SPA Static Assets and Fallback (Single Entity Deployment)
frontend_dist_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "dist"
)

if os.path.exists(frontend_dist_dir):
    assets_dir = os.path.join(frontend_dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API endpoints, docs, and websocket routes to pass through untouched
        if full_path.startswith(("api", "docs", "openapi.json", "redoc")):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")

        file_path = os.path.join(frontend_dist_dir, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            from fastapi.responses import FileResponse
            return FileResponse(file_path)

        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(frontend_dist_dir, "index.html"))
else:
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
