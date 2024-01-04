import numpy as np
from fusion import FusionAnalyzer

# Dummy but realistic inputs
gradcam = np.random.rand(128, 128)
patch_map = np.random.rand(128, 128)

ocr_data = [
    {"text": "PAY", "confidence": 90, "bbox": [10, 10, 100, 40]},
    {"text": "SALARY", "confidence": 88, "bbox": [120, 10, 260, 40]},
]


uncertainty = {
    "mean_probability": 0.29,
    "uncertainty": 0.03,
    "manual_review": False
}

# ✅ THIS LINE WAS MISSING OR MISNAMED
fusion = FusionAnalyzer()

# Positional arguments (as required by analyze())
result = fusion.analyze(
    gradcam,
    patch_map,
    ocr_data,
    uncertainty
)

print("Fusion result:")
print(result)
