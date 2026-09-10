"""Document type classification module.

Classifies documents into types (paystub, bank statement, ID, passport, etc.)
to enable type-specific validation rules and fusion weight adjustments.

Usage:
    from document_type_classifier import DocumentTypeClassifier
    classifier = DocumentTypeClassifier()
    
    # Rule-based classification (works without training)
    doc_type = classifier.classify_by_keywords(ocr_text)
    
    # CNN-based classification (requires trained model)
    doc_type = classifier.classify_by_cnn(image)
"""
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image


class DocumentTypeClassifier:
    """Classify document types for type-specific validation.
    
    Supports two classification methods:
    1. Keyword-based (OCR text): Fast, interpretable, no model needed
    2. CNN-based (image): More accurate, requires training
    
    Document types:
    - paystub: Pay stub / salary slip
    - bank_statement: Bank account statement
    - tax_form: Tax documents (W-2, 1099, etc.)
    - id_card: Government ID card
    - passport: Passport document
    - invoice: Business invoice
    - check: Bank check
    - other: Unclassified document
    """
    
    # Keyword signatures for each document type
    KEYWORD_SIGNATURES = {
        "paystub": [
            "gross pay", "net pay", "pay period", "salary", "wages",
            "deductions", "tax withholding", "year to date", "ytd",
            "employee", "employer", "pay stub", "paycheck",
            "federal tax", "state tax", "social security", "medicare",
        ],
        "bank_statement": [
            "account number", "statement period", "opening balance",
            "closing balance", "transaction", "deposit", "withdrawal",
            "bank statement", "account summary", "routing number",
            "available balance", "current balance",
        ],
        "tax_form": [
            "w-2", "w2", "1099", "tax return", "irs", "internal revenue",
            "taxable income", "adjusted gross", "form 1040", "filing status",
            "wages tips", "social security wages",
        ],
        "id_card": [
            "driver license", "driver's license", "identification",
            "date of birth", "expiry date", "issuing authority",
            "state id", "national id",
        ],
        "passport": [
            "passport", "nationality", "place of birth",
            "date of issue", "date of expiry", "passport number",
            "machine readable", "travel document",
        ],
        "invoice": [
            "invoice", "bill to", "ship to", "payment terms",
            "subtotal", "tax amount", "total due", "invoice number",
            "purchase order", "unit price", "quantity",
        ],
        "check": [
            "pay to the order", "memo", "routing number",
            "check number", "amount", "dollars",
        ],
    }
    
    # Type-specific fusion weight adjustments
    TYPE_WEIGHT_PROFILES = {
        "paystub": {
            "visual_anomaly": 0.20,
            "visual_ocr_overlap": 0.35,  # Text tampering is critical for paystubs
            "ocr_visual_conflict": 0.20,
            "low_ocr_confidence": 0.10,
            "uncertainty_penalty": 0.05,
            "spatial_density": 0.10,
        },
        "bank_statement": {
            "visual_anomaly": 0.20,
            "visual_ocr_overlap": 0.35,
            "ocr_visual_conflict": 0.20,
            "low_ocr_confidence": 0.10,
            "uncertainty_penalty": 0.05,
            "spatial_density": 0.10,
        },
        "id_card": {
            "visual_anomaly": 0.35,  # Visual forensics more important for IDs
            "visual_ocr_overlap": 0.25,
            "ocr_visual_conflict": 0.15,
            "low_ocr_confidence": 0.10,
            "uncertainty_penalty": 0.10,
            "spatial_density": 0.05,
        },
        "default": {
            "visual_anomaly": 0.25,
            "visual_ocr_overlap": 0.30,
            "ocr_visual_conflict": 0.15,
            "low_ocr_confidence": 0.10,
            "uncertainty_penalty": 0.10,
            "spatial_density": 0.10,
        },
    }
    
    def __init__(self, cnn_model=None):
        """Initialize document type classifier.
        
        Args:
            cnn_model: Optional trained CNN model for image-based classification.
                If None, only keyword-based classification is available.
        """
        self.cnn_model = cnn_model
    
    def classify_by_keywords(self, ocr_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify document type using OCR text keywords.
        
        Args:
            ocr_text: Full OCR-extracted text from the document.
            
        Returns:
            Tuple of (document_type, confidence, all_scores).
        """
        text_lower = ocr_text.lower()
        scores = {}
        
        for doc_type, keywords in self.KEYWORD_SIGNATURES.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = matches / len(keywords) if keywords else 0.0
        
        # Get best match
        if not scores or max(scores.values()) == 0:
            return "other", 0.0, scores
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence, scores
    
    def classify_by_keywords_from_results(self, ocr_results: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, float]]:
        """Classify document type from structured OCR results.
        
        Args:
            ocr_results: List of OCR result dicts with 'text' keys.
            
        Returns:
            Tuple of (document_type, confidence, all_scores).
        """
        combined_text = " ".join(r.get("text", "") for r in ocr_results)
        return self.classify_by_keywords(combined_text)
    
    def classify_by_cnn(self, image: Image.Image) -> Tuple[str, float]:
        """Classify document type using a trained CNN model.
        
        Args:
            image: PIL Image of the document.
            
        Returns:
            Tuple of (document_type, confidence).
            
        Raises:
            RuntimeError: If no CNN model is loaded.
        """
        if self.cnn_model is None:
            raise RuntimeError("No CNN model loaded for image-based classification. "
                             "Use classify_by_keywords() instead.")
        
        # Preprocess image
        img_array = np.array(image.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Predict
        preds = self.cnn_model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        
        # Map index to type
        type_list = list(self.KEYWORD_SIGNATURES.keys()) + ["other"]
        doc_type = type_list[class_idx] if class_idx < len(type_list) else "other"
        
        return doc_type, confidence
    
    def classify(self, image: Image.Image, 
                 ocr_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Comprehensive classification using all available methods.
        
        Args:
            image: PIL Image of the document.
            ocr_results: Optional OCR results for keyword classification.
            
        Returns:
            Dict with classification results from all methods.
        """
        result = {"methods": {}}
        
        # Keyword-based classification
        if ocr_results:
            doc_type, confidence, all_scores = self.classify_by_keywords_from_results(ocr_results)
            result["methods"]["keywords"] = {
                "type": doc_type,
                "confidence": confidence,
                "all_scores": all_scores,
            }
        
        # CNN-based classification
        if self.cnn_model is not None:
            try:
                doc_type, confidence = self.classify_by_cnn(image)
                result["methods"]["cnn"] = {
                    "type": doc_type,
                    "confidence": confidence,
                }
            except Exception:
                pass
        
        # Final decision: prefer CNN if available, fallback to keywords
        if "cnn" in result["methods"] and result["methods"]["cnn"]["confidence"] > 0.5:
            result["type"] = result["methods"]["cnn"]["type"]
            result["confidence"] = result["methods"]["cnn"]["confidence"]
        elif "keywords" in result["methods"] and result["methods"]["keywords"]["confidence"] > 0.1:
            result["type"] = result["methods"]["keywords"]["type"]
            result["confidence"] = result["methods"]["keywords"]["confidence"]
        else:
            result["type"] = "other"
            result["confidence"] = 0.0
        
        # Get type-specific weight profile
        result["weight_profile"] = self.get_weight_profile(result["type"])
        
        return result
    
    def get_weight_profile(self, doc_type: str) -> Dict[str, float]:
        """Get fusion weight profile for a document type.
        
        Args:
            doc_type: Document type string.
            
        Returns:
            Dict of signal name to weight.
        """
        return self.TYPE_WEIGHT_PROFILES.get(doc_type, self.TYPE_WEIGHT_PROFILES["default"])
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported document types."""
        return list(self.KEYWORD_SIGNATURES.keys()) + ["other"]
