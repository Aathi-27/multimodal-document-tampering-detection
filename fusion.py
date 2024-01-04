"""Cross-modal forensic fusion for document tampering detection.

This module combines multiple independent modalities (vision-based tamper detection,
OCR text extraction, uncertainty estimation) into a unified risk assessment.

Why rule-based fusion instead of learned fusion:
1. Interpretability: Each rule maps to forensic reasoning (e.g., "text altered" vs "background altered")
2. Data efficiency: Learned fusion requires large labeled datasets with ground-truth fusion targets
3. Causality: Rules encode domain knowledge about how tampering manifests across modalities
4. Debugging: Easy to trace which rule triggered high risk; learned models are black-box
5. Trust: Forensic applications require human-auditable decisions; rules provide transparency
6. Robustness: Rules generalize to unseen tampering patterns; learned models may overfit artifacts

Future work: If sufficient labeled data becomes available, rules can be replaced with
learned weights or used as features in a meta-classifier.

Usage:
    from fusion import FusionAnalyzer
    analyzer = FusionAnalyzer()
    result = analyzer.analyze(gradcam_heatmap, patch_heatmap, ocr_results, uncertainty_data)
    print(result['risk_tier'], result['risk_score'], result['explanations'])
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import json


class FusionAnalyzer:
    """Rule-based cross-modal fusion for forensic tamper detection.
    
    Combines visual anomaly detection (Grad-CAM, patch localization),
    text extraction (OCR), and uncertainty estimation (MC Dropout) into
    a unified risk assessment with human-readable explanations.
    """
    
    def __init__(self, 
                 visual_threshold: float = 0.5,
                 ocr_confidence_threshold: int = 60,
                 uncertainty_threshold: float = 0.15,
                 overlap_iou_threshold: float = 0.3):
        """Initialize fusion analyzer with decision thresholds.
        
        Args:
            visual_threshold: Minimum normalized heatmap value (0-1) to consider region suspicious.
            ocr_confidence_threshold: Minimum OCR confidence (0-100) to trust text detection.
            uncertainty_threshold: Maximum MC Dropout std dev to trust single-pass prediction.
            overlap_iou_threshold: Minimum IoU between visual anomaly and OCR box to flag overlap.
        """
        self.visual_threshold = visual_threshold
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.overlap_iou_threshold = overlap_iou_threshold
    
    def analyze(self, 
                gradcam_heatmap: np.ndarray,
                patch_heatmap: np.ndarray,
                ocr_results: List[Dict[str, Any]],
                uncertainty_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal fusion and compute risk assessment.
        
        Args:
            gradcam_heatmap: Grad-CAM heatmap array (H, W), normalized [0, 1].
            patch_heatmap: Patch localization heatmap (H, W), normalized [0, 1].
            ocr_results: List of dicts with keys 'text', 'confidence', 'bbox' [x1, y1, x2, y2].
            uncertainty_data: Dict with keys 'mean_probability', 'uncertainty_std', 'manual_review_required'.
        
        Returns:
            Dict with keys:
            - 'risk_score': float in [0, 1], higher = more suspicious
            - 'risk_tier': str in ['LOW', 'MEDIUM', 'HIGH']
            - 'explanations': list of str, human-readable reasons for risk assessment
            - 'details': dict with per-rule scores for transparency
        """
        # SPATIAL ALIGNMENT: Multimodal signals originate at different spatial resolutions.
        # Align all heatmaps and masks to the patch heatmap resolution (canonical grid).
        canonical_shape = patch_heatmap.shape  # (H, W)
        
        # Resize Grad-CAM heatmap to match patch heatmap resolution
        if gradcam_heatmap.shape != canonical_shape:
            # Use bilinear interpolation for continuous heatmaps to preserve intensity gradations
            gradcam_heatmap = cv2.resize(gradcam_heatmap, (canonical_shape[1], canonical_shape[0]), 
                                        interpolation=cv2.INTER_LINEAR)
        
        # Normalize OCR bounding boxes to canonical resolution
        # OCR boxes are in original image coordinates; scale them to match patch heatmap resolution
        ocr_results = self._normalize_ocr_bboxes(ocr_results, canonical_shape)
        
        explanations = []
        risk_components = {}
        
        # Rule 1: Visual anomaly intensity (average of Grad-CAM and patch localization)
        # Rationale: Both heatmaps detect pixel-level anomalies; averaging reduces false positives.
        visual_score = self._compute_visual_anomaly_score(gradcam_heatmap, patch_heatmap)
        risk_components['visual_anomaly'] = visual_score
        if visual_score > self.visual_threshold:
            explanations.append(f"Visual anomaly detected: {visual_score:.2f} exceeds threshold {self.visual_threshold}")
        
        # Rule 2: Visual-OCR overlap (text region tampering)
        # Rationale: If high tamper score overlaps text bounding boxes, text likely forged/altered.
        overlap_score, overlap_details = self._compute_visual_ocr_overlap(
            patch_heatmap, ocr_results)
        risk_components['visual_ocr_overlap'] = overlap_score
        if overlap_score > 0:
            explanations.append(
                f"Visual tamper overlaps {overlap_details['num_overlapping_boxes']} OCR text regions "
                f"(mean IoU: {overlap_details['mean_iou']:.2f}); possible text manipulation"
            )
        
        # Rule 3: High-confidence OCR in suspicious regions
        # Rationale: High OCR confidence + high visual tamper = text was cleanly altered (sophisticated forgery).
        ocr_visual_conflict = self._compute_ocr_visual_conflict(
            patch_heatmap, ocr_results)
        risk_components['ocr_visual_conflict'] = ocr_visual_conflict
        if ocr_visual_conflict > 0:
            explanations.append(
                f"High OCR confidence in visually suspicious regions: {ocr_visual_conflict:.2f}; "
                "text may be cleanly forged (sophisticated tampering)"
            )
        
        # Rule 4: Low OCR confidence overall
        # Rationale: Low OCR confidence may indicate blurred/distorted text from poor forgery or compression artifacts.
        low_ocr_confidence_score = self._compute_low_ocr_confidence(ocr_results)
        risk_components['low_ocr_confidence'] = low_ocr_confidence_score
        if low_ocr_confidence_score > 0:
            explanations.append(
                f"Low average OCR confidence: {low_ocr_confidence_score:.2f}; "
                "possible text distortion from tampering or poor image quality"
            )
        
        # Rule 5: Uncertainty penalty (MC Dropout)
        # Rationale: High uncertainty means model is unsure; reduces trust in automated decision.
        uncertainty_penalty = self._compute_uncertainty_penalty(uncertainty_data)
        risk_components['uncertainty_penalty'] = uncertainty_penalty
        if uncertainty_data.get('manual_review_required', False):
            explanations.append(
                f"High prediction uncertainty: {uncertainty_data['uncertainty_std']:.3f} "
                f"exceeds threshold {uncertainty_data['uncertainty_threshold']}; manual review required"
            )
        
        # Rule 6: Spatial density (multiple modalities agree)
        # Rationale: If Grad-CAM, patch localization, and OCR all flag same region, confidence increases.
        spatial_density = self._compute_spatial_density(
            gradcam_heatmap, patch_heatmap, ocr_results)
        risk_components['spatial_density'] = spatial_density
        if spatial_density > 0.6:
            explanations.append(
                f"Multiple modalities agree on suspicious region (density: {spatial_density:.2f}); "
                "strong evidence of localized tampering"
            )
        
        # Aggregate risk score: weighted sum of components
        # Weights chosen to prioritize visual-OCR overlap (most forensically relevant) and uncertainty.
        risk_score = (
            0.25 * visual_score +
            0.30 * overlap_score +
            0.15 * ocr_visual_conflict +
            0.10 * low_ocr_confidence_score +
            0.10 * uncertainty_penalty +
            0.10 * spatial_density
        )
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Tier assignment: LOW < 0.3, MEDIUM 0.3-0.6, HIGH >= 0.6
        if risk_score < 0.3:
            risk_tier = "LOW"
        elif risk_score < 0.6:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"
        
        # Add summary explanation if no specific rules triggered
        if not explanations:
            explanations.append(f"No significant anomalies detected across modalities; risk score: {risk_score:.2f}")
        
        return {
            'risk_score': float(risk_score),
            'risk_tier': risk_tier,
            'explanations': explanations,
            'details': risk_components
        }
    
    def _compute_visual_anomaly_score(self, gradcam: np.ndarray, patch: np.ndarray) -> float:
        """Compute average visual anomaly score from Grad-CAM and patch localization.
        
        Both heatmaps detect pixel-level anomalies but with different receptive fields:
        - Grad-CAM: global attention from final conv layer
        - Patch localization: local sliding-window scores
        Averaging reduces false positives from single-modality noise.
        """
        if gradcam.size == 0 or patch.size == 0:
            return 0.0
        gradcam_mean = np.mean(gradcam)
        patch_mean = np.mean(patch)
        return float((gradcam_mean + patch_mean) / 2.0)
    
    def _normalize_ocr_bboxes(self, ocr_results: List[Dict[str, Any]], 
                             canonical_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Normalize OCR bounding boxes to canonical heatmap resolution.
        
        OCR bounding boxes are in original image coordinates (unbounded).
        Scale them to match the canonical shape (H, W) used by all heatmaps.
        
        Args:
            ocr_results: List of dicts with 'bbox': [x1, y1, x2, y2] in original image coords.
            canonical_shape: Tuple (H, W) of the canonical heatmap resolution.
        
        Returns:
            List of dicts with bboxes scaled to canonical resolution.
        """
        h, w = canonical_shape
        normalized_results = []
        
        for item in ocr_results:
            normalized_item = item.copy()
            bbox = item['bbox']
            x1, y1, x2, y2 = bbox
            
            # Assume OCR bounding boxes are in original image coordinates.
            # Scale to canonical heatmap resolution.
            # (This assumes OCR was computed on the full original image,
            # and patch heatmap is the spatial reference for all modalities.)
            x1_norm = max(0, int(x1 * w / max(w, 1)))
            y1_norm = max(0, int(y1 * h / max(h, 1)))
            x2_norm = min(w, int(x2 * w / max(w, 1)))
            y2_norm = min(h, int(y2 * h / max(h, 1)))
            
            normalized_item['bbox'] = [x1_norm, y1_norm, x2_norm, y2_norm]
            normalized_results.append(normalized_item)
        
        return normalized_results
    
    def _compute_visual_ocr_overlap(self, heatmap: np.ndarray, 
                                   ocr_results: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Compute overlap between visual tamper regions and OCR bounding boxes.
        
        If high tamper scores coincide with text regions, text is likely forged.
        Uses IoU (Intersection over Union) to measure spatial overlap.
        All inputs are already aligned to the same spatial resolution.
        """
        if len(ocr_results) == 0 or heatmap.size == 0:
            return 0.0, {'num_overlapping_boxes': 0, 'mean_iou': 0.0}
        
        h, w = heatmap.shape
        overlapping_boxes = 0
        iou_scores = []
        
        for ocr_item in ocr_results:
            bbox = ocr_item['bbox']
            x1, y1, x2, y2 = bbox
            
            # Bboxes are already normalized to [0, w] and [0, h]; just clip bounds
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract heatmap region under OCR box
            region = heatmap[int(y1):int(y2), int(x1):int(x2)]
            if region.size == 0:
                continue
            
            # Compute mean tamper score in OCR region
            region_score = np.mean(region)
            
            # IoU-like metric: fraction of high-tamper pixels in OCR region
            high_tamper_fraction = np.sum(region > self.visual_threshold) / region.size
            
            if high_tamper_fraction > self.overlap_iou_threshold:
                overlapping_boxes += 1
                iou_scores.append(high_tamper_fraction)
        
        if overlapping_boxes == 0:
            return 0.0, {'num_overlapping_boxes': 0, 'mean_iou': 0.0}
        
        overlap_score = overlapping_boxes / len(ocr_results)  # fraction of OCR boxes overlapping tamper
        mean_iou = float(np.mean(iou_scores))
        
        return float(overlap_score), {'num_overlapping_boxes': overlapping_boxes, 'mean_iou': mean_iou}
    
    def _compute_ocr_visual_conflict(self, heatmap: np.ndarray,
                                     ocr_results: List[Dict[str, Any]]) -> float:
        """Detect high OCR confidence in visually suspicious regions.
        
        Rationale: If OCR is confident but visual tamper is high, text was likely
        cleanly forged (sophisticated attack). Naive forgeries show low OCR confidence.
        All inputs are already aligned to the same spatial resolution.
        """
        if len(ocr_results) == 0 or heatmap.size == 0:
            return 0.0
        
        h, w = heatmap.shape
        conflicts = []
        
        for ocr_item in ocr_results:
            confidence = ocr_item['confidence']
            if confidence < self.ocr_confidence_threshold:
                continue  # only consider high-confidence detections
            
            bbox = ocr_item['bbox']
            x1, y1, x2, y2 = bbox
            
            # Bboxes are already normalized; just clip bounds
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            region = heatmap[int(y1):int(y2), int(x1):int(x2)]
            if region.size == 0:
                continue
            
            region_score = np.mean(region)
            
            # Conflict: high OCR confidence + high visual tamper
            if region_score > self.visual_threshold:
                conflicts.append(region_score * (confidence / 100.0))
        
        if len(conflicts) == 0:
            return 0.0
        
        return float(np.mean(conflicts))
    
    def _compute_low_ocr_confidence(self, ocr_results: List[Dict[str, Any]]) -> float:
        """Compute penalty for low overall OCR confidence.
        
        Low confidence may indicate blurred/distorted text from poor forgery,
        compression artifacts, or low image quality.
        """
        if len(ocr_results) == 0:
            return 0.0
        
        confidences = [item['confidence'] for item in ocr_results]
        mean_confidence = np.mean(confidences)
        
        # Normalize: 0 confidence -> score 1.0, threshold confidence -> score 0.0
        if mean_confidence < self.ocr_confidence_threshold:
            return float(1.0 - (mean_confidence / self.ocr_confidence_threshold))
        return 0.0
    
    def _compute_uncertainty_penalty(self, uncertainty_data: Dict[str, Any]) -> float:
        """Compute penalty from MC Dropout uncertainty.
        
        High uncertainty reduces trust in automated decision; requires human review.
        """
        uncertainty = uncertainty_data.get('uncertainty_std', 0.0)
        threshold = uncertainty_data.get('uncertainty_threshold', self.uncertainty_threshold)
        
        if uncertainty >= threshold:
            # Normalize: threshold -> 0.5, 2*threshold -> 1.0
            return float(np.clip(uncertainty / threshold / 2.0, 0.0, 1.0))
        return 0.0
    
    def _compute_spatial_density(self, gradcam: np.ndarray, patch: np.ndarray,
                                 ocr_results: List[Dict[str, Any]]) -> float:
        """Compute spatial agreement between modalities.
        
        If Grad-CAM, patch localization, and OCR all flag the same region,
        evidence is stronger (multiple independent detectors agree).
        All inputs are already aligned to the same spatial resolution.
        """
        if gradcam.size == 0 or patch.size == 0:
            return 0.0
        
        # Threshold both heatmaps to create binary masks
        gradcam_mask = (gradcam > self.visual_threshold).astype(float)
        patch_mask = (patch > self.visual_threshold).astype(float)
        
        # Create OCR mask (regions with text bounding boxes)
        h, w = patch.shape
        ocr_mask = np.zeros((h, w), dtype=float)
        for ocr_item in ocr_results:
            bbox = ocr_item['bbox']
            x1, y1, x2, y2 = bbox
            
            # Bboxes are already normalized; just clip bounds
            x1 = max(0, int(x1))
            x2 = min(w, int(x2))
            y1 = max(0, int(y1))
            y2 = min(h, int(y2))
            
            if x2 > x1 and y2 > y1:
                ocr_mask[y1:y2, x1:x2] = 1.0
        
        # Compute intersection: regions where all three modalities agree
        intersection = gradcam_mask * patch_mask * ocr_mask
        density = np.sum(intersection) / max(np.sum(gradcam_mask + patch_mask + ocr_mask > 0), 1)
        
        return float(density)
    
    def save_results(self, analysis_result: Dict[str, Any], output_path: str) -> None:
        """Save fusion analysis results to JSON file.
        
        Provides human-readable risk assessment with full transparency into
        each rule's contribution and reasoning.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
