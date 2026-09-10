"""Multi-language OCR support module.

Extends the base OCR module with automatic language detection and
multi-language support for international documents.

Usage:
    from multi_language_ocr import MultiLanguageOCR
    ocr = MultiLanguageOCR()
    results = ocr.extract_with_auto_language(image)
    # or specify language:
    results = ocr.extract(image, language="eng+fra")
"""
import pytesseract
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple


class MultiLanguageOCR:
    """Multi-language OCR with automatic language detection.
    
    Supports:
    - Auto-detection of document language
    - Multiple languages in a single document
    - Fallback to English if detection fails
    - Language-specific post-processing
    """
    
    # Common document languages and their Tesseract codes
    SUPPORTED_LANGUAGES = {
        "english": "eng",
        "spanish": "spa",
        "french": "fra",
        "german": "deu",
        "italian": "ita",
        "portuguese": "por",
        "dutch": "nld",
        "russian": "rus",
        "chinese_simplified": "chi_sim",
        "chinese_traditional": "chi_tra",
        "japanese": "jpn",
        "korean": "kor",
        "arabic": "ara",
        "hindi": "hin",
        "tamil": "tam",
        "telugu": "tel",
        "bengali": "ben",
    }
    
    def __init__(self, confidence_threshold: int = 40, default_language: str = "eng"):
        """Initialize multi-language OCR.
        
        Args:
            confidence_threshold: Minimum OCR confidence (0-100).
            default_language: Fallback language if detection fails.
        """
        self.confidence_threshold = confidence_threshold
        self.default_language = default_language
    
    def get_available_languages(self) -> List[str]:
        """Get list of Tesseract language packs installed on the system."""
        try:
            langs = pytesseract.get_languages()
            return [l for l in langs if l != "osd"]  # Exclude orientation script detector
        except Exception:
            return ["eng"]
    
    def detect_language(self, image: Image.Image, sample_size: int = 500) -> str:
        """Auto-detect document language using a quick initial OCR pass.
        
        Uses Tesseract's OSD (Orientation and Script Detection) mode
        to identify the primary script/language in the document.
        
        Args:
            image: PIL Image to analyze.
            sample_size: Max width for quick sample (speed optimization).
            
        Returns:
            Tesseract language code (e.g., "eng", "fra").
        """
        try:
            # Quick OCR with OSD to detect script
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            script = osd.get("script", "").lower()
            
            # Map script names to language codes
            script_map = {
                "latin": "eng",
                "cyrillic": "rus",
                "han": "chi_sim",
                "hangul": "kor",
                "japanese": "jpn",
                "arabic": "ara",
                "devanagari": "hin",
                "tamil": "tam",
                "telugu": "tel",
                "bengali": "ben",
            }
            
            detected = script_map.get(script, self.default_language)
            
            # Verify the detected language pack is available
            available = self.get_available_languages()
            if detected in available:
                return detected
            
            return self.default_language
            
        except Exception:
            return self.default_language
    
    def extract(self, image: Image.Image, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract text with specified or default language.
        
        Args:
            image: PIL Image to extract text from.
            language: Tesseract language code(s), e.g., "eng" or "eng+fra".
                If None, uses default_language.
                
        Returns:
            List of dicts with 'text', 'confidence', 'bbox' keys.
        """
        if language is None:
            language = self.default_language
        
        try:
            data = pytesseract.image_to_data(
                image, lang=language, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR not found. Install: apt-get install tesseract-ocr")
        
        results = []
        n = len(data.get("text", []))
        
        for i in range(n):
            text = data["text"][i].strip() if data["text"][i] else ""
            
            try:
                confidence = int(float(data["conf"][i]))
            except (ValueError, TypeError):
                confidence = -1
            
            if not text or confidence < self.confidence_threshold:
                continue
            
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            
            results.append({
                "text": text,
                "confidence": confidence,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            })
        
        return results
    
    def extract_with_auto_language(self, image: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
        """Extract text with automatic language detection.
        
        First detects the document language, then runs OCR with that language.
        
        Args:
            image: PIL Image to extract text from.
            
        Returns:
            Tuple of (results_list, detected_language_code).
        """
        detected_lang = self.detect_language(image)
        results = self.extract(image, language=detected_lang)
        return results, detected_lang
    
    def extract_multilingual(self, image: Image.Image, 
                            languages: List[str]) -> List[Dict[str, Any]]:
        """Extract text from a multilingual document.
        
        Combines multiple language packs for documents containing
        text in multiple languages (e.g., English + French).
        
        Args:
            image: PIL Image to extract text from.
            languages: List of language codes, e.g., ["eng", "fra"].
            
        Returns:
            List of extracted text results.
        """
        # Combine language codes with + separator
        lang_string = "+".join(languages)
        
        # Verify all languages are available
        available = set(self.get_available_languages())
        valid_langs = [l for l in languages if l in available]
        
        if not valid_langs:
            valid_langs = [self.default_language]
        
        lang_string = "+".join(valid_langs)
        return self.extract(image, language=lang_string)
    
    @staticmethod
    def get_language_name(code: str) -> str:
        """Convert Tesseract language code to human-readable name."""
        reverse_map = {
            "eng": "English", "spa": "Spanish", "fra": "French",
            "deu": "German", "ita": "Italian", "por": "Portuguese",
            "nld": "Dutch", "rus": "Russian", "chi_sim": "Chinese (Simplified)",
            "chi_tra": "Chinese (Traditional)", "jpn": "Japanese",
            "kor": "Korean", "ara": "Arabic", "hin": "Hindi",
            "tam": "Tamil", "tel": "Telugu", "ben": "Bengali",
        }
        return reverse_map.get(code, code)
