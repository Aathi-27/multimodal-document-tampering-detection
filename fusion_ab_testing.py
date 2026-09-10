"""A/B testing framework for fusion weight profiles.

Allows configuring multiple fusion weight profiles, tracking their performance,
and selecting the best-performing profile over time.

Usage:
    from fusion_ab_testing import FusionABTester
    tester = FusionABTester("fusion_profiles.json")
    
    # Get a weight profile for this request (A/B split)
    profile_name, weights = tester.get_profile()
    
    # After getting ground truth, record the result
    tester.record_result(profile_name, correct=True, risk_score=0.85)
    
    # Get performance report
    report = tester.get_performance_report()
    tester.print_report(report)
"""
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


class FusionABTester:
    """A/B testing framework for fusion weight profiles.
    
    Features:
    - Define multiple weight profiles (A, B, C, ...)
    - Traffic splitting (e.g., 50/50 or 70/30)
    - Track accuracy per profile
    - Auto-select best profile
    - Persist results to JSON file
    """
    
    # Default weight profiles
    DEFAULT_PROFILES = {
        "baseline": {
            "weights": {
                "visual_anomaly": 0.25,
                "visual_ocr_overlap": 0.30,
                "ocr_visual_conflict": 0.15,
                "low_ocr_confidence": 0.10,
                "uncertainty_penalty": 0.10,
                "spatial_density": 0.10,
            },
            "traffic_share": 0.5,
            "description": "Original balanced weights",
        },
        "ocr_heavy": {
            "weights": {
                "visual_anomaly": 0.15,
                "visual_ocr_overlap": 0.40,
                "ocr_visual_conflict": 0.20,
                "low_ocr_confidence": 0.10,
                "uncertainty_penalty": 0.05,
                "spatial_density": 0.10,
            },
            "traffic_share": 0.25,
            "description": "Emphasizes OCR-based signals for text-heavy documents",
        },
        "visual_heavy": {
            "weights": {
                "visual_anomaly": 0.35,
                "visual_ocr_overlap": 0.25,
                "ocr_visual_conflict": 0.10,
                "low_ocr_confidence": 0.05,
                "uncertainty_penalty": 0.15,
                "spatial_density": 0.10,
            },
            "traffic_share": 0.25,
            "description": "Emphasizes visual forensics for image-based tampering",
        },
    }
    
    def __init__(self, storage_path: str = "fusion_ab_results.json",
                 profiles: Optional[Dict[str, Any]] = None):
        """Initialize A/B tester.
        
        Args:
            storage_path: Path to persist results JSON.
            profiles: Optional custom weight profiles. Uses DEFAULT_PROFILES if None.
        """
        self.storage_path = storage_path
        self.profiles = profiles or self.DEFAULT_PROFILES
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load persisted results from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Initialize empty results for each profile
        results = {}
        for name in self.profiles:
            results[name] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "unlabeled_predictions": 0,
                "risk_scores": [],
                "last_updated": None,
            }
        return results
    
    def _save_results(self):
        """Persist results to disk."""
        with open(self.storage_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def get_profile(self) -> Tuple[str, Dict[str, float]]:
        """Get a weight profile for the current request (traffic splitting).
        
        Uses weighted random selection based on traffic_share values.
        
        Returns:
            Tuple of (profile_name, weights_dict).
        """
        profile_names = list(self.profiles.keys())
        shares = [self.profiles[name].get("traffic_share", 1.0 / len(profile_names))
                  for name in profile_names]
        
        # Normalize shares
        total = sum(shares)
        shares = [s / total for s in shares]
        
        # Weighted random selection
        selected = random.choices(profile_names, weights=shares, k=1)[0]
        
        return selected, self.profiles[selected]["weights"]
    
    def record_result(self, profile_name: str, correct: Optional[bool],
                      risk_score: float, ground_truth: Optional[int] = None):
        """Record the outcome of a prediction using a specific profile.
        
        Args:
            profile_name: Name of the profile used.
            correct: Whether the prediction was correct (True/False/None if unknown).
            risk_score: The computed risk score.
            ground_truth: Optional ground truth label (0=clean, 1=tampered).
        """
        if profile_name not in self.results:
            self.results[profile_name] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "unlabeled_predictions": 0,
                "risk_scores": [],
                "last_updated": None,
            }
        
        self.results[profile_name]["total_predictions"] += 1
        self.results[profile_name]["risk_scores"].append(risk_score)
        
        if correct is True:
            self.results[profile_name]["correct_predictions"] += 1
        elif correct is False:
            self.results[profile_name]["incorrect_predictions"] += 1
        else:
            self.results[profile_name]["unlabeled_predictions"] += 1
        
        self.results[profile_name]["last_updated"] = datetime.now().isoformat()
        self._save_results()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all profiles.
        
        Returns:
            Dict with per-profile accuracy and statistics.
        """
        report = {"profiles": {}, "best_profile": None, "best_accuracy": 0.0}
        
        for name, data in self.results.items():
            labeled = data["correct_predictions"] + data["incorrect_predictions"]
            accuracy = data["correct_predictions"] / labeled if labeled > 0 else None
            
            avg_risk = float(sum(data["risk_scores"]) / len(data["risk_scores"])) if data["risk_scores"] else 0.0
            
            report["profiles"][name] = {
                "total_predictions": data["total_predictions"],
                "labeled_predictions": labeled,
                "accuracy": accuracy,
                "correct": data["correct_predictions"],
                "incorrect": data["incorrect_predictions"],
                "avg_risk_score": avg_risk,
                "description": self.profiles.get(name, {}).get("description", ""),
            }
            
            if accuracy is not None and accuracy > report["best_accuracy"]:
                report["best_accuracy"] = accuracy
                report["best_profile"] = name
        
        return report
    
    def get_best_profile(self) -> Tuple[str, Dict[str, float]]:
        """Get the best-performing profile based on historical accuracy.
        
        Returns:
            Tuple of (profile_name, weights_dict).
            Falls back to baseline if insufficient data.
        """
        report = self.get_performance_report()
        best = report.get("best_profile")
        
        if best and best in self.profiles:
            return best, self.profiles[best]["weights"]
        
        # Fallback to baseline
        return "baseline", self.profiles.get("baseline", self.DEFAULT_PROFILES["baseline"])["weights"]
    
    def print_report(self, report: Optional[Dict[str, Any]] = None):
        """Print a formatted performance report."""
        if report is None:
            report = self.get_performance_report()
        
        print("=" * 70)
        print("FUSION WEIGHT A/B TEST REPORT")
        print("=" * 70)
        
        for name, data in report["profiles"].items():
            acc_str = f"{data['accuracy']:.1%}" if data["accuracy"] is not None else "N/A"
            print(f"\n📊 Profile: {name}")
            print(f"   Description: {data['description']}")
            print(f"   Total Predictions: {data['total_predictions']}")
            print(f"   Labeled: {data['labeled_predictions']}")
            print(f"   Accuracy: {acc_str}")
            print(f"   Correct/Incorrect: {data['correct']}/{data['incorrect']}")
            print(f"   Avg Risk Score: {data['avg_risk_score']:.3f}")
        
        print(f"\n🏆 Best Profile: {report['best_profile']} ({report['best_accuracy']:.1%})")
        print("=" * 70)
    
    def reset(self):
        """Reset all results."""
        self.results = {}
        for name in self.profiles:
            self.results[name] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "unlabeled_predictions": 0,
                "risk_scores": [],
                "last_updated": None,
            }
        self._save_results()
