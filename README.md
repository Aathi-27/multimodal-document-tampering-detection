# multimodal-document-tampering-detection

Real-time bank document fraud detection using a 6-signal CV+OCR fusion pipeline. Instead of relying on Error Level Analysis alone, this system combines six complementary signals — ELA, Grad-CAM explainability, Monte Carlo Dropout uncertainty, OCR semantic conflict, OCR extraction confidence, and spatial IoU-style patch overlap — into a single weighted risk score, served as an AWS SageMaker real-time inference API.

---

## Architecture Overview

The pipeline is split into six modular analyzers that each produce one normalized scalar signal, which are then fused by a weighted combiner into a final tiered risk output.

```
Document Image
      │
      ├─► ela.py               → ELA tampering score + heatmap
      ├─► grad_cam.py          → Grad-CAM saliency score + heatmap
      ├─► mc_dropout.py        → MC Dropout confidence / epistemic risk scalar
      ├─► ocr.py               → OCR semantic conflict score + OCR extraction confidence score
      ├─► patch_localization.py→ Spatial density / IoU-style overlap score
      │
      └─► fusion.py            → Weighted risk score → Low / Medium / High risk band
```

---

## Module Breakdown

### `ela.py` — Error Level Analysis
Runs Error Level Analysis on the uploaded document image to expose compression artifacts left behind by tampering. The image is re-saved at a known JPEG quality level and differenced against the original; anomalous regions appear as bright patches. Outputs:
- **ELA heatmap** — pixel-level visualization of compression anomalies
- **ELA tampering score** — normalized aggregate intensity of anomalous regions across the page

### `grad_cam.py` — Gradient-weighted Class Activation Mapping
Wraps the EfficientNetB7-based CNN classifier with Grad-CAM to localize which spatial regions drive the model's tampering vs. clean prediction. Gradients of the target class are pooled over the final convolutional feature maps to produce a saliency map. Outputs:
- **Grad-CAM heatmap** — spatial saliency overlay showing model-relevant regions
- **Grad-CAM saliency score** — scalar reflecting how strongly suspicious regions influence the classification decision

### `mc_dropout.py` — Monte Carlo Dropout Uncertainty
Applies MC Dropout at inference time by keeping dropout layers active and running multiple stochastic forward passes through the CNN head. The variance across passes is used to estimate predictive (epistemic) uncertainty. Outputs:
- **MC Dropout confidence / risk scalar** — rewards stable, high-confidence predictions; penalizes high-variance uncertain predictions

### `ocr.py` — OCR Semantic Validation
Runs OCR over the document and aligns extracted text against the expected document schema (fields such as account number, totals, dates, signatures). Produces two independent scores:
- **OCR semantic conflict score** — captures mismatches between extracted text and business rules (e.g., inconsistent totals, out-of-range values, missing mandatory fields)
- **OCR extraction confidence score** — measures how reliably text was read from the image (penalizes blur, artifacts, and low-resolution regions)

### `patch_localization.py` — Spatial Density and IoU-style Overlap
Segments the page into patches and computes local anomaly density from ELA and Grad-CAM outputs. Measures spatial agreement between visual anomaly clusters and semantically important OCR fields (signatures, amounts, account numbers, dates) using an IoU-style overlap metric. Outputs:
- **Spatial density / overlap score** — high when suspicious pixels concentrate around semantically important document regions rather than background

### `fusion.py` — Weighted Signal Fusion
Takes the six normalized scalars and computes a weighted risk score:

| Signal | Source | Weight |
|---|---|---|
| ELA tampering score | ela.py | 0.25 |
| Grad-CAM saliency score | grad_cam.py | 0.30 |
| MC Dropout confidence/risk | mc_dropout.py | 0.10 |
| OCR semantic conflict score | ocr.py | 0.15 |
| OCR extraction confidence score | ocr.py | 0.10 |
| Spatial density / overlap score | patch_localization.py | 0.10 |

Visual forensics (ELA + Grad-CAM, combined weight 0.55) dominate the final score. OCR semantic conflict adds a strong independent signal. Uncertainty, confidence, and spatial overlap modulate the score rather than driving it alone.

Final output:
- **Continuous fraud risk score** in [0, 1]
- **Discrete risk band**: Low / Medium / High for downstream workflow integration

---

## Model and Framework

- **Deep learning framework**: TensorFlow / Keras
- **Base model**: EfficientNetB7 pretrained on ImageNet, fine-tuned on ELA-transformed document crops (tampering vs. clean labels)
- **Training and deployment**: Amazon SageMaker — Studio notebooks for training, real-time HTTPS inference endpoint for serving
- **Explainability**: Grad-CAM over the final convolutional block
- **Uncertainty**: Monte Carlo Dropout at inference (dropout kept active, N stochastic forward passes)
- **OCR engine**: Integrated via `ocr.py` for text extraction and schema validation

---

## End-to-End Inference Flow

1. Upload a document image to the SageMaker real-time endpoint
2. `ela.py` computes ELA heatmap and tampering score
3. `grad_cam.py` runs EfficientNetB7 forward pass, computes Grad-CAM saliency score
4. `mc_dropout.py` runs N stochastic forward passes, computes confidence/risk scalar
5. `ocr.py` extracts text, validates against schema, emits semantic conflict + extraction confidence scores
6. `patch_localization.py` computes patch anomaly density and IoU-style overlap with OCR fields
7. `fusion.py` applies weighted combination → continuous risk score + Low/Medium/High risk band
8. Response returned as JSON with risk score, risk band, ELA heatmap, and Grad-CAM heatmap

---

## Repository Structure

```
├── ela.py                        # Error Level Analysis module
├── grad_cam.py                   # Grad-CAM explainability module
├── mc_dropout.py                 # Monte Carlo Dropout uncertainty module
├── ocr.py                        # OCR extraction and semantic validation module
├── patch_localization.py         # Spatial patch density and IoU overlap module
├── fusion.py                     # Weighted signal fusion and risk scoring
├── model/                        # Trained EfficientNetB7 model artifacts
├── images/                       # Sample document images for testing
├── tampering_detection_training.ipynb   # SageMaker training notebook
└── tampering_detection_model_deploy.ipynb # SageMaker deployment notebook
```

---

## License

MIT-0
