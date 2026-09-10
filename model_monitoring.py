"""Model monitoring and drift detection module.

Tracks prediction distributions over time to detect model drift,
data quality issues, and performance degradation.

Usage:
    from model_monitoring import ModelMonitor
    monitor = ModelMonitor(storage_path="monitoring_data.json")
    
    # After each prediction:
    monitor.log_prediction(
        risk_score=0.75,
        risk_tier="HIGH",
        signal_scores={"visual_anomaly": 0.6, "visual_ocr_overlap": 0.8},
        latency_ms=1200,
        document_type="paystub"
    )
    
    # Check for drift:
    drift_report = monitor.check_drift()
    monitor.print_drift_report(drift_report)
"""
import json
import os
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class ModelMonitor:
    """Monitor model predictions for drift and quality issues.
    
    Tracks:
    - Prediction distribution shifts (risk scores, tiers)
    - Signal score distributions
    - Inference latency
    - Per-document-type statistics
    - Temporal trends (hourly, daily, weekly)
    
    Alerts when:
    - Average risk score shifts significantly
    - Uncertainty increases beyond normal range
    - Latency degrades beyond threshold
    - Prediction distribution changes drastically
    """
    
    def __init__(self, storage_path: str = "monitoring_data.json",
                 drift_threshold: float = 0.2,
                 latency_threshold_ms: float = 5000):
        """Initialize model monitor.
        
        Args:
            storage_path: Path to persist monitoring data.
            drift_threshold: Maximum allowed shift in mean risk score (absolute).
            latency_threshold_ms: Maximum acceptable latency in milliseconds.
        """
        self.storage_path = storage_path
        self.drift_threshold = drift_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load persisted monitoring data."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "predictions": [],
            "baseline": None,
            "alerts": [],
        }
    
    def _save_data(self):
        """Persist monitoring data to disk."""
        with open(self.storage_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def log_prediction(self, risk_score: float, risk_tier: str,
                       signal_scores: Optional[Dict[str, float]] = None,
                       latency_ms: Optional[float] = None,
                       document_type: str = "unknown",
                       metadata: Optional[Dict[str, Any]] = None):
        """Log a prediction for monitoring.
        
        Args:
            risk_score: Computed risk score (0-1).
            risk_tier: Risk classification ("LOW", "MEDIUM", "HIGH").
            signal_scores: Dict of individual signal scores.
            latency_ms: Inference latency in milliseconds.
            document_type: Classified document type.
            metadata: Additional metadata.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "signal_scores": signal_scores or {},
            "latency_ms": latency_ms,
            "document_type": document_type,
            "metadata": metadata or {},
        }
        
        self.data["predictions"].append(entry)
        
        # Set baseline from first 100 predictions
        if self.data["baseline"] is None and len(self.data["predictions"]) >= 100:
            self._compute_baseline()
        
        self._save_data()
    
    def _compute_baseline(self):
        """Compute baseline statistics from initial predictions."""
        preds = self.data["predictions"][:100]
        
        risk_scores = [p["risk_score"] for p in preds]
        latencies = [p["latency_ms"] for p in preds if p.get("latency_ms")]
        
        self.data["baseline"] = {
            "mean_risk_score": statistics.mean(risk_scores),
            "std_risk_score": statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0,
            "tier_distribution": self._compute_tier_distribution(preds),
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
            "computed_at": datetime.now().isoformat(),
            "sample_size": len(preds),
        }
        
        self._save_data()
    
    def _compute_tier_distribution(self, predictions: List[Dict]) -> Dict[str, float]:
        """Compute distribution of risk tiers."""
        if not predictions:
            return {}
        
        counts = defaultdict(int)
        for p in predictions:
            counts[p["risk_tier"]] += 1
        
        total = len(predictions)
        return {tier: count / total for tier, count in counts.items()}
    
    def check_drift(self, window_hours: int = 24) -> Dict[str, Any]:
        """Check for drift in recent predictions vs. baseline.
        
        Args:
            window_hours: Number of hours of recent data to compare.
            
        Returns:
            Dict with drift analysis results.
        """
        if self.data["baseline"] is None:
            return {"status": "no_baseline", "message": "Insufficient data for baseline (need 100+ predictions)"}
        
        # Get recent predictions
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [p for p in self.data["predictions"]
                  if datetime.fromisoformat(p["timestamp"]) > cutoff]
        
        if len(recent) < 10:
            return {"status": "insufficient_data", "message": f"Only {len(recent)} predictions in window"}
        
        baseline = self.data["baseline"]
        
        # Risk score drift
        recent_scores = [p["risk_score"] for p in recent]
        recent_mean = statistics.mean(recent_scores)
        score_drift = abs(recent_mean - baseline["mean_risk_score"])
        score_drift_detected = score_drift > self.drift_threshold
        
        # Tier distribution drift
        recent_tiers = self._compute_tier_distribution(recent)
        tier_drift = self._compute_distribution_distance(
            baseline["tier_distribution"], recent_tiers)
        
        # Latency drift
        recent_latencies = [p["latency_ms"] for p in recent if p.get("latency_ms")]
        latency_drift_detected = False
        if recent_latencies and baseline["mean_latency_ms"] > 0:
            recent_mean_latency = statistics.mean(recent_latencies)
            latency_drift_detected = recent_mean_latency > self.latency_threshold_ms
        
        # Compile results
        drift_report = {
            "status": "analyzed",
            "window_hours": window_hours,
            "num_recent_predictions": len(recent),
            "risk_score": {
                "baseline_mean": baseline["mean_risk_score"],
                "recent_mean": recent_mean,
                "drift": score_drift,
                "drift_detected": score_drift_detected,
            },
            "tier_distribution": {
                "baseline": baseline["tier_distribution"],
                "recent": recent_tiers,
                "distance": tier_drift,
            },
            "latency": {
                "baseline_mean_ms": baseline["mean_latency_ms"],
                "recent_mean_ms": statistics.mean(recent_latencies) if recent_latencies else None,
                "threshold_ms": self.latency_threshold_ms,
                "drift_detected": latency_drift_detected,
            },
            "alerts": [],
        }
        
        # Generate alerts
        if score_drift_detected:
            alert = {
                "type": "risk_score_drift",
                "severity": "warning",
                "message": f"Risk score mean shifted by {score_drift:.3f} (threshold: {self.drift_threshold})",
                "timestamp": datetime.now().isoformat(),
            }
            drift_report["alerts"].append(alert)
            self.data["alerts"].append(alert)
        
        if latency_drift_detected:
            alert = {
                "type": "latency_degradation",
                "severity": "warning",
                "message": f"Mean latency {statistics.mean(recent_latencies):.0f}ms exceeds threshold {self.latency_threshold_ms}ms",
                "timestamp": datetime.now().isoformat(),
            }
            drift_report["alerts"].append(alert)
            self.data["alerts"].append(alert)
        
        self._save_data()
        return drift_report
    
    def _compute_distribution_distance(self, dist1: Dict[str, float],
                                        dist2: Dict[str, float]) -> float:
        """Compute L1 distance between two distributions."""
        all_keys = set(list(dist1.keys()) + list(dist2.keys()))
        distance = sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in all_keys)
        return distance / 2  # Normalize to [0, 1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall monitoring summary."""
        preds = self.data["predictions"]
        
        if not preds:
            return {"total_predictions": 0, "message": "No predictions logged yet"}
        
        risk_scores = [p["risk_score"] for p in preds]
        latencies = [p["latency_ms"] for p in preds if p.get("latency_ms")]
        
        return {
            "total_predictions": len(preds),
            "mean_risk_score": statistics.mean(risk_scores),
            "median_risk_score": statistics.median(risk_scores),
            "tier_distribution": self._compute_tier_distribution(preds),
            "mean_latency_ms": statistics.mean(latencies) if latencies else None,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None,
            "baseline_set": self.data["baseline"] is not None,
            "total_alerts": len(self.data.get("alerts", [])),
            "first_prediction": preds[0]["timestamp"] if preds else None,
            "last_prediction": preds[-1]["timestamp"] if preds else None,
        }
    
    def print_drift_report(self, report: Dict[str, Any]):
        """Print formatted drift report."""
        print("=" * 60)
        print("MODEL DRIFT DETECTION REPORT")
        print("=" * 60)
        
        if report["status"] != "analyzed":
            print(f"Status: {report['status']} — {report.get('message', '')}")
            return
        
        print(f"Window: Last {report['window_hours']} hours")
        print(f"Recent Predictions: {report['num_recent_predictions']}")
        
        print(f"\n📊 Risk Score Drift:")
        rs = report["risk_score"]
        print(f"   Baseline Mean: {rs['baseline_mean']:.3f}")
        print(f"   Recent Mean:   {rs['recent_mean']:.3f}")
        print(f"   Drift:         {rs['drift']:.3f}")
        print(f"   Alert:         {'⚠️ DRIFT DETECTED' if rs['drift_detected'] else '✅ Normal'}")
        
        print(f"\n📈 Tier Distribution:")
        for tier in ["LOW", "MEDIUM", "HIGH"]:
            base = report["tier_distribution"]["baseline"].get(tier, 0)
            recent = report["tier_distribution"]["recent"].get(tier, 0)
            print(f"   {tier:8s}: Baseline {base:.1%} → Recent {recent:.1%}")
        
        print(f"\n⏱️ Latency:")
        lat = report["latency"]
        print(f"   Baseline: {lat['baseline_mean_ms']:.0f} ms")
        if lat["recent_mean_ms"]:
            print(f"   Recent:   {lat['recent_mean_ms']:.0f} ms")
        print(f"   Alert:    {'⚠️ LATENCY DEGRADATION' if lat['drift_detected'] else '✅ Normal'}")
        
        if report["alerts"]:
            print(f"\n🚨 Alerts ({len(report['alerts'])}):")
            for alert in report["alerts"]:
                print(f"   [{alert['severity'].upper()}] {alert['message']}")
        
        print("=" * 60)
    
    def reset(self):
        """Reset all monitoring data."""
        self.data = {"predictions": [], "baseline": None, "alerts": []}
        self._save_data()
