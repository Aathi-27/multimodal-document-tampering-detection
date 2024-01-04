"""OCR (Optical Character Recognition) extraction module.

This module extracts text and metadata from document images using Tesseract.
OCR is treated as an independent modality that provides complementary information
about document content, distinct from vision-based tamper detection.

Why OCR is kept separate:
- Vision (Grad-CAM, patch localization, MC Dropout) detects pixel-level anomalies
- OCR detects content-level information (what text is present, where, confidence)
- These are orthogonal signals: vision finds where image was altered, OCR finds what text exists
- Early separation allows independent validation and easier debugging
- Fusion can happen later with explicit rules/learned weights
- Allows OCR to fail gracefully without affecting vision pipeline

Usage:
    from ocr import OCRExtractor
    extractor = OCRExtractor()
    results = extractor.extract(image)
    viz = extractor.visualize_boxes(image, results)
    viz.save("ocr_boxes.png")
"""
import json
from typing import List, Dict, Any
import pytesseract
from PIL import Image, ImageDraw


class OCRExtractor:
    """OCR text extraction and visualization using Tesseract.
    
    Extracts text tokens, bounding boxes, and confidence scores from images.
    Provides visualization overlays for validation.
    """
    
    def __init__(self, confidence_threshold: int = 40):
        """Initialize OCR extractor.
        
        Args:
            confidence_threshold: Minimum OCR confidence (0-100) to include token.
                Lower threshold includes more text; higher filters uncertain detections.
        """
        if not (0 <= confidence_threshold <= 100):
            raise ValueError("confidence_threshold must be in [0, 100].")
        self.confidence_threshold = confidence_threshold
    
    def extract(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text tokens and metadata from image via Tesseract.
        
        Uses pytesseract to call Tesseract OCR engine, extracting words/tokens,
        their bounding boxes, and confidence scores.
        
        Args:
            image: PIL Image (RGB or grayscale) to extract text from.
            
        Returns:
            List of dicts, each with keys:
            - 'text': extracted text token (str)
            - 'confidence': OCR confidence 0-100 (int)
            - 'bbox': bounding box as [x1, y1, x2, y2] in pixels (list of int)
            
        Raises:
            RuntimeError: If Tesseract binary not found on system PATH.
        """
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract OCR binary not found. Install via: "
                "Windows: choco install tesseract, "
                "Linux: apt-get install tesseract-ocr, "
                "macOS: brew install tesseract"
            ) from e
        
        results = []
        n = len(data.get("text", []))
        
        for i in range(n):
            text = data["text"][i].strip() if data["text"][i] else ""
            
            # Parse confidence score; default to -1 if parsing fails
            try:
                confidence = int(float(data["conf"][i]))
            except (ValueError, TypeError):
                confidence = -1
            
            # Filter by confidence threshold
            if not text or confidence < self.confidence_threshold:
                continue
            
            # Extract bounding box
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = [int(x), int(y), int(x + w), int(y + h)]  # [x1, y1, x2, y2]
            
            results.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })
        
        return results
    
    def visualize_boxes(self, image: Image.Image, ocr_results: List[Dict[str, Any]], 
                       box_color: str = "red", width: int = 2) -> Image.Image:
        """Draw OCR bounding boxes on image.
        
        Creates a copy of the image with rectangles drawn around detected text regions.
        Useful for validation: visually confirm that OCR found the expected text areas.
        
        Args:
            image: PIL Image (RGB) to draw boxes on.
            ocr_results: List of dicts from extract(), each with 'bbox' key.
            box_color: Color for box outlines (e.g., 'red', 'blue', '#FF0000').
            width: Line width for box outlines in pixels.
            
        Returns:
            New PIL Image with boxes drawn (does not modify input image).
        """
        # Create copy in RGB mode
        viz = image.copy().convert("RGB")
        draw = ImageDraw.Draw(viz)
        
        for result in ocr_results:
            x1, y1, x2, y2 = result["bbox"]
            # Draw rectangle: outline only (no fill)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width)
        
        return viz
    
    def save_results(self, ocr_results: List[Dict[str, Any]], output_path: str) -> None:
        """Save OCR results to JSON file.
        
        Structured JSON format allows downstream processing to read extracted text
        and validate OCR quality independently from vision signals.
        
        Args:
            ocr_results: List of dicts from extract().
            output_path: Path to write JSON file (e.g., 'results.json').
        """
        output_dict = {
            "ocr_engine": "tesseract",
            "confidence_threshold": self.confidence_threshold,
            "num_tokens": len(ocr_results),
            "results": ocr_results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
    
    def extract_text_only(self, image: Image.Image) -> str:
        """Extract full text from image as single string (convenience method).
        
        Ignores spatial information; useful for downstream NLP tasks.
        
        Args:
            image: PIL Image to extract text from.
            
        Returns:
            Concatenated text from all detected tokens (str).
        """
        results = self.extract(image)
        return " ".join([r["text"] for r in results])
