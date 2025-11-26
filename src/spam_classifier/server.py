"""
FastAPI server for spam classification
"""

from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config
from .model import SpamClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Spam/Ham Classifier API",
    description="NLP-based email spam classification system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier: Optional[SpamClassifier] = None


# Request/Response models
class PredictRequest(BaseModel):
    """Request model for prediction"""

    text: str = Field(..., min_length=1, description="Email text to classify")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won a free iPhone. Click here to claim now!"
            }
        }


class PredictResponse(BaseModel):
    """Response model for prediction"""

    label: str = Field(..., description="Predicted label: 'ham' or 'spam'")
    confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence (0-1)"
    )
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    is_spam: bool = Field(..., description="Boolean flag for spam detection")
    cleaned_text: str = Field(..., description="Preprocessed text")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "spam",
                "confidence": 0.9876,
                "probabilities": {"ham": 0.0124, "spam": 0.9876},
                "is_spam": True,
                "cleaned_text": "congratulation win free iphone click claim",
            }
        }


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction"""

    texts: List[str] = Field(..., min_items=1, max_items=100)

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Hey, are we still meeting for lunch?",
                    "FREE PRIZE! Click now to win $1000!",
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    version: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global classifier

    try:
        print("Loading spam classifier model...")
        classifier = SpamClassifier()
        classifier.load()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load model: {e}")
        print("  Server will start but predictions will fail until model is trained.")


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    web_dir = Path(__file__).parent.parent.parent / "web"
    index_path = web_dir / "index.html"

    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Spam Classifier API</title></head>
                <body style="font-family: Arial; padding: 50px; text-align: center;">
                    <h1>🚀 Spam Classifier API</h1>
                    <p>API is running! Visit <a href="/api/docs">/api/docs</a> for interactive documentation.</p>
                    <p>Web UI not found. Check the <code>web/</code> directory.</p>
                </body>
            </html>
            """,
            status_code=200,
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.model is not None,
        "version": "1.0.0",
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict spam/ham for a single text

    - **text**: Email or message text to classify

    Returns prediction with confidence scores
    """
    if classifier is None or classifier.model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    try:
        result = classifier.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/batch", response_model=List[PredictResponse])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict spam/ham for multiple texts

    - **texts**: List of email/message texts (max 100)

    Returns list of predictions
    """
    if classifier is None or classifier.model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400, detail="Maximum 100 texts allowed per batch request"
        )

    try:
        results = classifier.predict_batch(request.texts)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/api/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """
    Predict spam/ham from uploaded text file

    Accepts .txt files with one message per line
    """
    if classifier is None or classifier.model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    try:
        content = await file.read()
        text = content.decode("utf-8")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if len(lines) > 100:
            raise HTTPException(
                status_code=400,
                detail="File contains more than 100 lines. Maximum 100 messages allowed.",
            )

        results = classifier.predict_batch(lines)

        return {
            "filename": file.filename,
            "total_messages": len(lines),
            "spam_count": sum(1 for r in results if r["is_spam"]),
            "ham_count": sum(1 for r in results if not r["is_spam"]),
            "predictions": results,
        }
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding error. Please use UTF-8 encoded text files.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


# Mount static files (for web UI)
web_static_dir = Path(__file__).parent.parent.parent / "web" / "static"
if web_static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_static_dir)), name="static")


def main():
    """Run the server"""
    uvicorn.run(
        "src.spam_classifier.server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
    )


if __name__ == "__main__":
    main()
