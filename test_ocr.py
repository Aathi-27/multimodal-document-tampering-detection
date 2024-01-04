from PIL import Image
from ocr import OCRExtractor
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load image
img = Image.open("images/predict/Paystub.jpg")

# OCR
ocr = OCRExtractor(confidence_threshold=40)
results = ocr.extract(img)

print("Number of OCR tokens:", len(results))

# Print a few samples
for r in results[:5]:
    print(r)
