# Multimodal Document Tampering Detection v2.0

Real-time bank document fraud detection using an **optimized 6-signal CV+OCR fusion pipeline** with parallel execution, batched inference, and a modern React dashboard.

---

## 🚀 What's New in v2.0

### Performance Optimizations (6× faster)
- **Batched patch localization** — 20× speedup (single `model.predict()` call instead of 165 sequential)
- **Batched MC Dropout** — 7.5× speedup (tiled input, single forward pass)
- **Parallel pipeline** — OCR, Grad-CAM, MC Dropout, and patch heatmap run concurrently
- **Cached Grad-CAM model** — Built once in `__init__`, not reconstructed per call
- **In-memory ELA** — `BytesIO` instead of disk I/O (4× faster, thread-safe)
- **Early-exit inference** — Skip expensive stages when model is confident (40-60% latency reduction)

### New Features
- **📄 PDF Support** — Upload and analyze PDF documents (via PyMuPDF)
- **🌐 FastAPI REST Backend** — Production-ready API with OpenAPI docs, CORS, API key auth
- **📊 React/Next.js Dashboard** — Modern analytics dashboard with upload, history, settings
- **📋 Report Generation** — HTML and JSON reports with embedded visualizations
- **🔍 Copy-Move Forgery Detection** — 7th signal for detecting cloned regions
- **🏷️ Document Type Classifier** — Auto-detect paystubs, bank statements, IDs, passports
- **⚖️ A/B Testing Framework** — Test fusion weight profiles and auto-select best performer
- **📈 Model Monitoring** — Track prediction drift, latency degradation, and alerting
- **🧪 Adversarial Robustness Testing** — Test against JPEG compression, noise, blur, etc.
- **🔐 Document Hashing** — SHA-256, pHash, dHash for integrity verification
- **🌐 Multi-Language OCR** — Auto-detect language, support for 17+ languages
- **🔔 Alerting System** — Email (SES), Slack, SMS (Twilio), webhook notifications
- **🗂️ Data Augmentation Pipeline** — Synthetic tampering for training data expansion
- **⚡ Model Optimization Scripts** — TFLite (INT8/FP16), ONNX export, benchmarking

---

## Architecture

```
Document Image / PDF
      │
      ├─► ela.py               → ELA tampering score + heatmap (in-memory)
      ├─► grad_cam.py          → Grad-CAM saliency score + heatmap (cached model)
      ├─► mc_dropout.py        → MC Dropout confidence (batched inference)
      ├─► ocr.py               → OCR semantic conflict + extraction confidence
      ├─► patch_localization.py→ Spatial density / overlap (batched patches)
      ├─► copy_move_detection.py → Copy-move forgery detection (new)
      ├─► document_type_classifier.py → Auto document type (new)
      │
      └─► fusion.py            → Weighted risk score → Low / Medium / High
            │
            ├─► fusion_ab_testing.py    → A/B test weight profiles
            ├─► model_monitoring.py     → Track drift and performance
            └─► alerting.py             → Send notifications
```

---

## Quick Start

### Streamlit App (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### FastAPI Backend
```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for Swagger UI
```

### React Dashboard
```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit UI with parallel pipeline |
| `api_server.py` | FastAPI REST backend |
| `ela.py` | Error Level Analysis (in-memory, multi-quality) |
| `grad_cam.py` | Grad-CAM explainability (cached model) |
| `mc_dropout.py` | MC Dropout uncertainty (batched) |
| `ocr.py` | OCR text extraction |
| `patch_localization.py` | Patch-level tamper heatmap (batched) |
| `fusion.py` | Weighted signal fusion |
| `report_generator.py` | HTML/JSON report generation |
| `document_hashing.py` | Perceptual and cryptographic hashing |
| `copy_move_detection.py` | Copy-move forgery detection |
| `adversarial_testing.py` | Robustness testing suite |
| `multi_language_ocr.py` | Multi-language OCR with auto-detection |
| `document_type_classifier.py` | Document type classification |
| `fusion_ab_testing.py` | A/B testing for fusion weights |
| `model_monitoring.py` | Prediction drift detection |
| `alerting.py` | Email/Slack/SMS/webhook alerts |
| `data_augmentation.py` | Training data augmentation |
| `model_optimization.py` | TFLite/ONNX conversion scripts |
| `dashboard/` | React/Next.js analytics dashboard |

---

## Configuration

Copy `.env.example` to `.env` and configure:
- API authentication
- Alert channels (email, Slack, SMS, webhook)
- CORS origins for the dashboard
- Alert thresholds

---

## Model Optimization

```bash
# TFLite INT8 (75% smaller, 3-4× faster)
python model_optimization.py --format tflite --output model_optimized/

# TFLite FP16 (50% smaller, 1.5-2× faster)
python model_optimization.py --format tflite_fp16 --output model_optimized/

# ONNX (2-3× faster, cross-platform)
python model_optimization.py --format onnx --output model_optimized/

# Benchmark
python model_optimization.py --format benchmark --model model/1
```

---

## License

MIT-0
