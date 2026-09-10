"""Adversarial robustness testing module.

Tests the document tampering detection model against common adversarial attacks
and perturbations to assess robustness. Reports accuracy degradation per attack type.

Usage:
    from adversarial_testing import AdversarialRobustnessTester
    tester = AdversarialRobustnessTester(model)
    report = tester.run_full_suite(test_images, test_labels)
    tester.print_report(report)
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, List, Any, Tuple, Optional
import io


class AdversarialRobustnessTester:
    """Test model robustness against common perturbations and adversarial attacks.
    
    Attack categories:
    1. Compression artifacts (JPEG at various qualities)
    2. Noise injection (Gaussian, salt-and-pepper, speckle)
    3. Blur (Gaussian, motion, median)
    4. Brightness/contrast changes
    5. Geometric (rotation, scaling, cropping)
    6. Print-scan simulation
    7. Color space manipulation
    """
    
    def __init__(self, model, input_size: Tuple[int, int] = (128, 128)):
        """Initialize adversarial tester.
        
        Args:
            model: Trained Keras/TensorFlow model.
            input_size: Model's expected input dimensions (H, W).
        """
        self.model = model
        self.input_size = input_size
    
    def run_full_suite(self, images: List[np.ndarray], labels: List[int],
                       preprocess_fn=None) -> Dict[str, Any]:
        """Run full adversarial robustness test suite.
        
        Args:
            images: List of preprocessed input arrays (N, H, W, 3).
            labels: List of ground truth labels (0=clean, 1=tampered).
            preprocess_fn: Optional function to preprocess images before perturbation.
            
        Returns:
            Dict with per-attack accuracy, degradation, and detailed results.
        """
        images = np.array(images)
        labels = np.array(labels)
        
        # Baseline accuracy
        baseline_preds = self._predict(images)
        baseline_acc = float(np.mean(baseline_preds == labels))
        
        results = {
            "baseline_accuracy": baseline_acc,
            "num_samples": len(labels),
            "attacks": {},
        }
        
        # Define attack suite
        attacks = {
            "jpeg_q90": lambda x: self._jpeg_compress(x, 90),
            "jpeg_q70": lambda x: self._jpeg_compress(x, 70),
            "jpeg_q50": lambda x: self._jpeg_compress(x, 50),
            "jpeg_q30": lambda x: self._jpeg_compress(x, 30),
            "gaussian_noise_low": lambda x: self._add_gaussian_noise(x, 0.01),
            "gaussian_noise_high": lambda x: self._add_gaussian_noise(x, 0.05),
            "salt_pepper_low": lambda x: self._add_salt_pepper(x, 0.01),
            "salt_pepper_high": lambda x: self._add_salt_pepper(x, 0.05),
            "gaussian_blur_light": lambda x: self._blur(x, radius=1),
            "gaussian_blur_heavy": lambda x: self._blur(x, radius=3),
            "brightness_up": lambda x: self._adjust_brightness(x, 1.3),
            "brightness_down": lambda x: self._adjust_brightness(x, 0.7),
            "contrast_up": lambda x: self._adjust_contrast(x, 1.5),
            "contrast_down": lambda x: self._adjust_contrast(x, 0.5),
            "rotation_5deg": lambda x: self._rotate(x, 5),
            "rotation_15deg": lambda x: self._rotate(x, 15),
            "scale_down_80pct": lambda x: self._scale(x, 0.8),
            "print_scan_sim": lambda x: self._print_scan_simulation(x),
            "color_desaturation": lambda x: self._desaturate(x),
        }
        
        for attack_name, attack_fn in attacks.items():
            perturbed = np.array([attack_fn(img) for img in images])
            perturbed_preds = self._predict(perturbed)
            accuracy = float(np.mean(perturbed_preds == labels))
            degradation = baseline_acc - accuracy
            
            results["attacks"][attack_name] = {
                "accuracy": accuracy,
                "degradation": degradation,
                "accuracy_drop_pct": (degradation / baseline_acc * 100) if baseline_acc > 0 else 0,
            }
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable robustness report.
        
        Args:
            results: Results dict from run_full_suite().
            
        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ADVERSARIAL ROBUSTNESS REPORT")
        lines.append("=" * 60)
        lines.append(f"Baseline Accuracy: {results['baseline_accuracy']:.3f}")
        lines.append(f"Samples Tested: {results['num_samples']}")
        lines.append("-" * 60)
        lines.append(f"{'Attack':<30} {'Accuracy':>10} {'Drop':>10} {'Drop%':>10}")
        lines.append("-" * 60)
        
        for name, data in sorted(results["attacks"].items(), 
                                  key=lambda x: x[1]["degradation"], reverse=True):
            lines.append(
                f"{name:<30} {data['accuracy']:>10.3f} "
                f"{data['degradation']:>10.3f} {data['accuracy_drop_pct']:>9.1f}%"
            )
        
        lines.append("-" * 60)
        
        # Summary
        avg_degradation = np.mean([d["degradation"] for d in results["attacks"].values()])
        max_attack = max(results["attacks"].items(), key=lambda x: x[1]["degradation"])
        
        lines.append(f"Average Degradation: {avg_degradation:.3f}")
        lines.append(f"Most Vulnerable: {max_attack[0]} ({max_attack[1]['degradation']:.3f} drop)")
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        return report
    
    def _predict(self, images: np.ndarray) -> np.ndarray:
        """Run batch prediction."""
        preds = self.model.predict(images, verbose=0)
        return np.argmax(preds, axis=1)
    
    # ── Perturbation Functions ──
    
    def _jpeg_compress(self, img: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression at given quality."""
        pil_img = self._array_to_pil(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        return self._pil_to_array(compressed)
    
    def _add_gaussian_noise(self, img: np.ndarray, variance: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, variance, img.shape)
        noisy = np.clip(img + noise, 0, 1)
        return noisy.astype(np.float32)
    
    def _add_salt_pepper(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Add salt-and-pepper noise."""
        result = img.copy()
        total_pixels = img.shape[0] * img.shape[1]
        num_salt = int(total_pixels * amount / 2)
        num_pepper = int(total_pixels * amount / 2)
        
        # Salt (white pixels)
        coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
        result[coords[0], coords[1]] = 1.0
        
        # Pepper (black pixels)
        coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
        result[coords[0], coords[1]] = 0.0
        
        return result
    
    def _blur(self, img: np.ndarray, radius: int = 2) -> np.ndarray:
        """Apply Gaussian blur."""
        pil_img = self._array_to_pil(img)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self._pil_to_array(blurred)
    
    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        pil_img = self._array_to_pil(img)
        enhanced = ImageEnhance.Brightness(pil_img).enhance(factor)
        return self._pil_to_array(enhanced)
    
    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        pil_img = self._array_to_pil(img)
        enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
        return self._pil_to_array(enhanced)
    
    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        pil_img = self._array_to_pil(img)
        rotated = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        return self._pil_to_array(rotated)
    
    def _scale(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Scale image down then back up (simulates resolution loss)."""
        pil_img = self._array_to_pil(img)
        h, w = pil_img.size[1], pil_img.size[0]
        small = pil_img.resize((int(w * factor), int(h * factor)), Image.BILINEAR)
        restored = small.resize((w, h), Image.BILINEAR)
        return self._pil_to_array(restored)
    
    def _print_scan_simulation(self, img: np.ndarray) -> np.ndarray:
        """Simulate print-scan artifacts: blur + noise + contrast change."""
        result = self._blur(img, radius=1)
        result = self._add_gaussian_noise(result, 0.02)
        result = self._adjust_contrast(result, 0.9)
        result = self._adjust_brightness(result, 1.05)
        return result
    
    def _desaturate(self, img: np.ndarray) -> np.ndarray:
        """Reduce color saturation."""
        pil_img = self._array_to_pil(img)
        desaturated = ImageEnhance.Color(pil_img).enhance(0.3)
        return self._pil_to_array(desaturated)
    
    # ── Helper Functions ──
    
    def _array_to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert normalized array to PIL Image."""
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr.astype(np.uint8))
    
    def _pil_to_array(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to normalized array."""
        img = img.resize(self.input_size).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
