"""FastAPI REST backend for document tampering detection.

Serves the optimized detection pipeline as a REST API with:
- Async processing with background tasks
- WebSocket for real-time progress updates
- OpenAPI/Swagger documentation
- API key authentication
- CORS support for React frontend

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    
    Or:
    python api_server.py
"""
import io
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with:\n"
        "  pip install fastapi uvicorn python-multipart"
    )

# ── App Initialization ──
app = FastAPI(
    title="Document Tampering Detection API",
    description="Real-time document fraud detection using 6-signal CV+OCR fusion pipeline",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (optional)
API_KEY = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key if configured."""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# ── Request/Response Models ──
class PredictionResponse(BaseModel):
    request_id: str
    risk_score: float
    risk_tier: str
    tamper_probability: float
    mc_dropout_mean: float
    mc_dropout_uncertainty: float
    explanations: List[str]
    ocr_token_count: int
    document_type: Optional[str] = None
    timing_ms: Dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class BatchPredictionRequest(BaseModel):
    request_ids: List[str]


# ── Lazy Model Loading ──
_model = None
_start_time = time.time()


def get_model():
    """Lazy-load model on first request."""
    global _model
    if _model is None:
        import tensorflow as tf
        model_path = os.path.join("model", "1")
        if os.path.exists(model_path):
            _model = tf.keras.models.load_model(model_path)
        else:
            raise RuntimeError(f"Model not found at {model_path}")
    return _model


# ── Pipeline Functions (optimized) ──
def process_document(image_bytes: bytes, early_exit: bool = True) -> Dict[str, Any]:
    """Process a document through the full detection pipeline."""
    import tensorflow as tf
    from concurrent.futures import ThreadPoolExecutor
    
    from ela import convert_to_ela_image
    from grad_cam import GradCAMExplainer
    from mc_dropout import MCDropoutUncertainty
    from patch_localization import PatchLocalizer
    from fusion import FusionAnalyzer
    from document_type_classifier import DocumentTypeClassifier
    
    timings = {}
    
    # Parse image
    t0 = time.time()
    orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    timings["image_load"] = (time.time() - t0) * 1000
    
    # ELA preprocessing (in-memory)
    t0 = time.time()
    ela_img = convert_to_ela_image(orig_img, quality=90)
    timings["ela"] = (time.time() - t0) * 1000
    
    # Prepare input tensor
    x = np.array(ela_img.resize((128, 128))).astype(np.float32) / 255.0
    x = x.reshape(1, 128, 128, 3)
    
    model = get_model()
    
    # Single forward pass
    t0 = time.time()
    preds = model(x, training=False).numpy()
    tamper_prob = float(preds[0][1])
    timings["single_pass"] = (time.time() - t0) * 1000
    
    # Early exit check
    is_confident = tamper_prob > 0.9 or tamper_prob < 0.1
    
    if early_exit and is_confident:
        timings["early_exit"] = True
        
        # Run GradCAM and OCR in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            def run_gradcam():
                explainer = GradCAMExplainer(model)
                return explainer.make_heatmap(x)
            
            def run_ocr():
                import pytesseract
                data = pytesseract.image_to_data(orig_img, output_type=pytesseract.Output.DICT)
                results = []
                for i in range(len(data.get("text", []))):
                    text = data["text"][i].strip() if data["text"][i] else ""
                    conf = int(float(data["conf"][i])) if data["conf"][i] else -1
                    if text and conf >= 40:
                        x1, y1, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        results.append({"text": text, "confidence": conf, "bbox": [int(x1), int(y1), int(x1+w), int(y1+h)]})
                return results
            
            fut_gc = pool.submit(run_gradcam)
            fut_ocr = pool.submit(run_ocr)
        
        gradcam_heatmap = fut_gc.result()
        ocr_results = fut_ocr.result()
        
        mc_mean = tamper_prob
        mc_uncertainty = 0.0
        patch_heatmap = np.zeros(orig_img.size[::-1], dtype=np.float32)
    else:
        timings["early_exit"] = False
        
        # Run all stages in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            def run_gradcam():
                explainer = GradCAMExplainer(model)
                return explainer.make_heatmap(x)
            
            def run_mc():
                estimator = MCDropoutUncertainty(model, num_samples=20)
                return estimator.estimate(x)
            
            def run_patch():
                localizer = PatchLocalizer(model, patch_size=128, stride=64)
                ela_arr = np.array(ela_img).astype(np.float32)
                if ela_arr.max() > 1.0:
                    ela_arr /= 255.0
                return localizer.localize(ela_arr)
            
            def run_ocr():
                import pytesseract
                data = pytesseract.image_to_data(orig_img, output_type=pytesseract.Output.DICT)
                results = []
                for i in range(len(data.get("text", []))):
                    text = data["text"][i].strip() if data["text"][i] else ""
                    conf = int(float(data["conf"][i])) if data["conf"][i] else -1
                    if text and conf >= 40:
                        x1, y1, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        results.append({"text": text, "confidence": conf, "bbox": [int(x1), int(y1), int(x1+w), int(y1+h)]})
                return results
            
            fut_gc = pool.submit(run_gradcam)
            fut_mc = pool.submit(run_mc)
            fut_patch = pool.submit(run_patch)
            fut_ocr = pool.submit(run_ocr)
        
        gradcam_heatmap = fut_gc.result()
        mc_mean, mc_uncertainty, _ = fut_mc.result()
        patch_heatmap = fut_patch.result()
        ocr_results = fut_ocr.result()
    
    # Document type classification
    doc_classifier = DocumentTypeClassifier()
    doc_type_result = doc_classifier.classify(orig_img, ocr_results)
    
    # Fusion
    t0 = time.time()
    analyzer = FusionAnalyzer()
    uncertainty_data = {
        "mean_probability": mc_mean,
        "uncertainty_std": mc_uncertainty,
        "uncertainty_threshold": 0.15,
        "manual_review_required": mc_uncertainty >= 0.15,
    }
    
    fusion_result = analyzer.analyze(gradcam_heatmap, patch_heatmap, ocr_results, uncertainty_data)
    timings["fusion"] = (time.time() - t0) * 1000
    
    total_time = sum(v for v in timings.values() if isinstance(v, (int, float)))
    timings["total"] = total_time
    
    return {
        "tamper_prob": tamper_prob,
        "mc_mean": mc_mean,
        "mc_uncertainty": mc_uncertainty,
        "risk_score": fusion_result["risk_score"],
        "risk_tier": fusion_result["risk_tier"],
        "explanations": fusion_result["explanations"],
        "ocr_results": ocr_results,
        "document_type": doc_type_result.get("type", "unknown"),
        "timings": timings,
        "fusion_details": fusion_result.get("details", {}),
    }


# ── API Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="2.0.0",
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_document(
    file: UploadFile = File(...),
    early_exit: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """Analyze a document for tampering.
    
    Upload a JPEG, PNG, or PDF document. Returns risk score,
    risk tier, signal scores, OCR results, and timing breakdown.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    # Read file
    contents = await file.read()
    
    if len(contents) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    # Handle PDF
    if file.content_type == "application/pdf":
        try:
            import fitz
            doc = fitz.open(stream=contents, filetype="pdf")
            page = doc[0]  # First page only
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            contents = img_data
            doc.close()
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF support requires PyMuPDF: pip install PyMuPDF")
    
    # Process
    request_id = str(uuid.uuid4())
    try:
        result = process_document(contents, early_exit=early_exit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    return PredictionResponse(
        request_id=request_id,
        risk_score=result["risk_score"],
        risk_tier=result["risk_tier"],
        tamper_probability=result["tamper_prob"],
        mc_dropout_mean=result["mc_mean"],
        mc_dropout_uncertainty=result["mc_uncertainty"],
        explanations=result["explanations"],
        ocr_token_count=len(result["ocr_results"]),
        document_type=result["document_type"],
        timing_ms=result["timings"],
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    early_exit: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """Analyze multiple documents in batch.
    
    Upload up to 10 documents. Returns results for each.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch request")
    
    results = []
    for file in files:
        contents = await file.read()
        request_id = str(uuid.uuid4())
        
        try:
            result = process_document(contents, early_exit=early_exit)
            results.append({
                "request_id": request_id,
                "filename": file.filename,
                "risk_score": result["risk_score"],
                "risk_tier": result["risk_tier"],
                "tamper_probability": result["tamper_prob"],
                "document_type": result["document_type"],
                "timing_ms": result["timings"],
                "status": "success",
            })
        except Exception as e:
            results.append({
                "request_id": request_id,
                "filename": file.filename,
                "status": "error",
                "error": str(e),
            })
    
    return {"batch_results": results, "total": len(results)}


@app.get("/report/{request_id}")
async def get_report(request_id: str):
    """Get a detailed report for a previous prediction.
    
    NOTE: Reports are stored in memory for demo. In production,
    use a database or S3 storage.
    """
    # Placeholder: in production, fetch from database
    raise HTTPException(status_code=404, detail="Report not found. Reports are not persisted in demo mode.")


@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get aggregate statistics about processed documents."""
    from model_monitoring import ModelMonitor
    monitor = ModelMonitor()
    return monitor.get_summary()


@app.get("/model/info")
async def model_info():
    """Get model architecture and metadata."""
    model = get_model()
    
    return {
        "architecture": "EfficientNetB7",
        "input_shape": [128, 128, 3],
        "num_classes": 2,
        "classes": ["clean", "tampered"],
        "total_params": model.count_params(),
        "trainable_params": sum(np.prod(w.shape) for w in model.trainable_weights),
        "framework": "TensorFlow/Keras",
    }


# ── WebSocket for real-time progress ──
@app.websocket("/ws/progress/{request_id}")
async def websocket_progress(websocket, request_id: str):
    """WebSocket endpoint for real-time progress updates during processing."""
    await websocket.accept()
    
    # Simulate progress updates (in production, hook into pipeline stages)
    stages = ["Loading model", "ELA preprocessing", "CNN inference", 
              "MC Dropout", "OCR extraction", "Patch localization", "Fusion"]
    
    for i, stage in enumerate(stages):
        await websocket.send_json({
            "request_id": request_id,
            "stage": stage,
            "progress": (i + 1) / len(stages),
            "timestamp": datetime.now().isoformat(),
        })
        import asyncio
        await asyncio.sleep(0.5)
    
    await websocket.send_json({
        "request_id": request_id,
        "stage": "complete",
        "progress": 1.0,
        "timestamp": datetime.now().isoformat(),
    })
    await websocket.close()


# ── Entry Point ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
