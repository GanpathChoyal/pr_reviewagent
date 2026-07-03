import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from api.reviews import router as reviews_router
from api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from repositories import db_manager
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app = FastAPI(
    title="PR Review Agent",
    description="Automated Pull Request Review Agent using Multi-Agent Architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.api_rate_limit
)
app.include_router(reviews_router)
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        db_manager.initialize()
        logging.info("Database initialized")
    except Exception as e:
        logging.warning(f"Database initialization skipped: {e}")
        logging.info("Running without database - reviews won't be persisted")
@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    await db_manager.close()
    logging.info("Database connections closed")
@app.get("/")
async def root():
    return {
        "message": "PR Review Agent API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected" if db_manager.engine else "not initialized"
    }
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)

