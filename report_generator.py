"""Document tampering detection report generator.

Generates comprehensive PDF and HTML reports for analyzed documents,
including all visualizations, scores, OCR results, and fusion explanations.

Usage:
    from report_generator import ReportGenerator
    generator = ReportGenerator()
    generator.generate_pdf(pipeline_results, original_image, output_path="report.pdf")
    generator.generate_html(pipeline_results, original_image, output_path="report.html")
"""
import json
import os
import io
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw


class ReportGenerator:
    """Generate tamper detection reports in PDF and HTML formats.
    
    Reports include:
    - Original image and all heatmap overlays
    - All signal scores and fusion explanation
    - OCR extracted text
    - Timing breakdown
    - Document hash for integrity verification
    - Timestamp and metadata
    """
    
    def __init__(self, company_name: str = "Document Tampering Detection System"):
        """Initialize report generator.
        
        Args:
            company_name: Name to display in report header.
        """
        self.company_name = company_name
    
    def _compute_document_hash(self, image: Image.Image) -> str:
        """Compute SHA-256 hash of document image for integrity verification."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()[:32]
    
    def _create_heatmap_overlay(self, base_image: Image.Image, heatmap: np.ndarray, 
                                 alpha: float = 0.4) -> Image.Image:
        """Create a heatmap overlay for report visualization."""
        import cv2
        base = base_image.convert("RGB")
        heat_resized = cv2.resize(heatmap, base.size)
        heat_clipped = np.clip(heat_resized, 0.0, 1.0).astype(np.float64)
        heat_uint8 = (heat_clipped * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay = np.array(base) * (1 - alpha) + heat_color * alpha
        return Image.fromarray(np.uint8(overlay))
    
    def generate_html(self, results: Dict[str, Any], original_image: Image.Image,
                      output_path: str = "report.html",
                      document_name: str = "Unknown Document") -> str:
        """Generate an HTML report.
        
        Args:
            results: Pipeline results dict from run_pipeline().
            original_image: Original PIL Image.
            output_path: Path to save HTML file.
            document_name: Name/identifier for the document.
            
        Returns:
            Path to generated HTML file.
        """
        doc_hash = self._compute_document_hash(original_image)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        fusion = results.get("fusion", {})
        risk_score = fusion.get("risk_score", 0)
        risk_tier = fusion.get("risk_tier", "UNKNOWN")
        explanations = fusion.get("explanations", [])
        details = fusion.get("details", {})
        
        tamper_prob = results.get("tamper_prob", 0)
        mc_mean = results.get("mc_mean", 0)
        mc_uncertainty = results.get("mc_uncertainty", 0)
        ocr_results = results.get("ocr_results", [])
        timings = results.get("timings", {})
        
        # Risk tier color
        risk_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
        risk_color = risk_colors.get(risk_tier, "#95a5a6")
        
        # Save images as base64 for embedding
        import base64
        def img_to_base64(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        
        orig_b64 = img_to_base64(original_image)
        
        patch_heatmap = results.get("patch_heatmap")
        heat_overlay_b64 = ""
        if patch_heatmap is not None and patch_heatmap.size > 10:
            heat_overlay = self._create_heatmap_overlay(original_image, patch_heatmap)
            heat_overlay_b64 = img_to_base64(heat_overlay)
        
        gradcam_heatmap = results.get("gradcam_heatmap")
        gradcam_b64 = ""
        if gradcam_heatmap is not None and hasattr(gradcam_heatmap, 'size') and gradcam_heatmap.size > 10:
            gradcam_overlay = self._create_heatmap_overlay(original_image, gradcam_heatmap)
            gradcam_b64 = img_to_base64(gradcam_overlay)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Tampering Detection Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 5px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
        .risk-badge {{ display: inline-block; padding: 8px 24px; border-radius: 20px; font-size: 18px; font-weight: bold; color: white; background: {risk_color}; margin: 10px 0; }}
        .card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .card h2 {{ font-size: 18px; margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        .scores-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .score-item {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .score-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .score-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; margin-top: 4px; }}
        .images-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
        .images-grid img {{ width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
        .explanation-list {{ list-style: none; }}
        .explanation-list li {{ padding: 10px 15px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #f39c12; border-radius: 0 8px 8px 0; font-size: 14px; }}
        .meta-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px; color: #7f8c8d; }}
        .ocr-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .ocr-table th, .ocr-table td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
        .ocr-table th {{ background: #f8f9fa; font-weight: 600; }}
        .timing-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }}
        .timing-item {{ text-align: center; padding: 10px; background: #e8f8f5; border-radius: 8px; }}
        .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
        @media print {{ body {{ background: white; }} .card {{ break-inside: avoid; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 {self.company_name}</h1>
            <div class="subtitle">Document Tampering Detection Report</div>
            <div style="margin-top:10px;">
                <span class="risk-badge">{risk_tier} RISK — Score: {risk_score:.3f}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>📋 Document Information</h2>
            <div class="meta-info">
                <div><strong>Document:</strong> {document_name}</div>
                <div><strong>Analyzed:</strong> {timestamp}</div>
                <div><strong>Document Hash (SHA-256):</strong> {doc_hash}...</div>
                <div><strong>Early Exit:</strong> {'Yes ⚡' if timings.get('early_exit') else 'No'}</div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Signal Scores</h2>
            <div class="scores-grid">
                <div class="score-item">
                    <div class="score-value">{tamper_prob:.3f}</div>
                    <div class="score-label">Tamper Probability</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{mc_mean:.3f}</div>
                    <div class="score-label">MC Dropout Mean</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{mc_uncertainty:.3f}</div>
                    <div class="score-label">Uncertainty (σ)</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{risk_score:.3f}</div>
                    <div class="score-label">Fusion Risk Score</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🖼️ Visual Analysis</h2>
            <div class="images-grid">
                <div>
                    <h3 style="font-size:14px;margin-bottom:8px;">Original Document</h3>
                    <img src="data:image/png;base64,{orig_b64}" alt="Original">
                </div>
                {f'<div><h3 style="font-size:14px;margin-bottom:8px;">Patch Heatmap Overlay</h3><img src="data:image/png;base64,{heat_overlay_b64}" alt="Heatmap"></div>' if heat_overlay_b64 else ''}
                {f'<div><h3 style="font-size:14px;margin-bottom:8px;">Grad-CAM Saliency</h3><img src="data:image/png;base64,{gradcam_b64}" alt="Grad-CAM"></div>' if gradcam_b64 else ''}
            </div>
        </div>
        
        <div class="card">
            <h2>🔎 Fusion Explanations</h2>
            <ul class="explanation-list">
                {''.join(f'<li>{exp}</li>' for exp in explanations) if explanations else '<li>No significant anomalies detected.</li>'}
            </ul>
        </div>
        
        <div class="card">
            <h2>📝 OCR Extracted Text ({len(ocr_results)} tokens)</h2>
            {'<table class="ocr-table"><tr><th>Text</th><th>Confidence</th><th>Bounding Box</th></tr>' + 
             ''.join(f'<tr><td>{r["text"]}</td><td>{r["confidence"]}</td><td>{r["bbox"]}</td></tr>' for r in ocr_results[:50]) +
             '</table>' if ocr_results else '<p>No text detected above confidence threshold.</p>'}
        </div>
        
        <div class="card">
            <h2>⏱️ Timing Breakdown</h2>
            <div class="timing-grid">
                {''.join(f'<div class="timing-item"><div class="score-value" style="font-size:18px;">{v:.3f}s</div><div class="score-label">{k}</div></div>' 
                         for k, v in timings.items() if isinstance(v, (int, float)))}
            </div>
        </div>
        
        <div class="footer">
            Generated by {self.company_name} — {timestamp}<br>
            This report is generated automatically and should be reviewed by qualified personnel.
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return output_path
    
    def generate_json(self, results: Dict[str, Any], original_image: Image.Image,
                      output_path: str = "report.json",
                      document_name: str = "Unknown Document") -> str:
        """Generate a JSON report for programmatic consumption.
        
        Args:
            results: Pipeline results dict.
            original_image: Original PIL Image (for hashing).
            output_path: Path to save JSON file.
            document_name: Name/identifier for the document.
            
        Returns:
            Path to generated JSON file.
        """
        doc_hash = self._compute_document_hash(original_image)
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "document_name": document_name,
                "document_hash_sha256": doc_hash,
                "generator_version": "2.0",
            },
            "scores": {
                "tamper_probability": results.get("tamper_prob", 0),
                "mc_dropout_mean": results.get("mc_mean", 0),
                "mc_dropout_uncertainty": results.get("mc_uncertainty", 0),
            },
            "fusion": results.get("fusion", {}),
            "ocr": {
                "num_tokens": len(results.get("ocr_results", [])),
                "tokens": results.get("ocr_results", []),
            },
            "timings": {k: v for k, v in results.get("timings", {}).items() 
                       if isinstance(v, (int, float, bool))},
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
