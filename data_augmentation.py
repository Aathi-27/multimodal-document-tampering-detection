"""Training data augmentation pipeline for document tampering detection.

Generates synthetic tampered documents and applies augmentations to expand
the training dataset. Supports both geometric and semantic augmentations.

Usage:
    from data_augmentation import DocumentAugmentor
    augmentor = DocumentAugmentor()
    
    # Augment a single image
    augmented = augmentor.augment(image, label=0)
    
    # Generate synthetic tampered document
    tampered = augmentor.synthesize_tamper(original_image)
    
    # Augment entire dataset
    augmentor.augment_dataset("images/training/", output_dir="images/augmented/")
"""
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Any


class DocumentAugmentor:
    """Data augmentation pipeline for document tampering detection.
    
    Augmentation categories:
    1. Geometric: rotation, perspective, scaling, cropping
    2. Quality: JPEG compression, blur, noise, brightness/contrast
    3. Semantic: text overlay, region copy-move, amount modification
    4. Print-scan simulation
    
    Goal: Expand training data from ~404 images to 2000+ samples.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize augmentor with random seed for reproducibility."""
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def augment(self, image: Image.Image, label: int = 0,
                num_augmentations: int = 5) -> List[Tuple[Image.Image, int]]:
        """Generate augmented versions of an image.
        
        Args:
            image: PIL Image to augment.
            label: Original label (0=clean, 1=tampered).
            num_augmentations: Number of augmented versions to generate.
            
        Returns:
            List of (augmented_image, label) tuples.
        """
        results = []
        
        augmentations = [
            self._random_rotation,
            self._random_brightness_contrast,
            self._random_noise,
            self._random_blur,
            self._jpeg_compression,
            self._random_crop_resize,
            self._perspective_transform,
            self._print_scan_simulation,
        ]
        
        for _ in range(num_augmentations):
            aug_fn = self.rng.choice(augmentations)
            augmented = aug_fn(image.copy())
            results.append((augmented, label))
        
        return results
    
    def synthesize_tamper(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Generate a synthetic tampered version of a clean document.
        
        Applies one of several tampering techniques:
        1. Region copy-move (copy a region and paste elsewhere)
        2. Text overlay (add/modify text)
        3. Region splicing (insert region from another image)
        4. Amount modification (modify numeric regions)
        
        Args:
            image: Clean PIL Image to tamper.
            
        Returns:
            Tuple of (tampered_image, tamper_type).
        """
        techniques = [
            self._copy_move_tamper,
            self._text_overlay_tamper,
            self._region_erase_tamper,
            self._brightness_patch_tamper,
        ]
        
        technique = self.rng.choice(techniques)
        tampered, tamper_type = technique(image.copy())
        
        return tampered, tamper_type
    
    # ── Geometric Augmentations ──
    
    def _random_rotation(self, image: Image.Image) -> Image.Image:
        """Random rotation between -5 and 5 degrees."""
        angle = self.rng.uniform(-5, 5)
        return image.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    
    def _random_crop_resize(self, image: Image.Image) -> Image.Image:
        """Random crop then resize back to original size."""
        w, h = image.size
        crop_ratio = self.rng.uniform(0.85, 0.98)
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)
        
        left = self.rng.randint(0, w - new_w)
        top = self.rng.randint(0, h - new_h)
        
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)
    
    def _perspective_transform(self, image: Image.Image) -> Image.Image:
        """Apply random perspective distortion."""
        w, h = image.size
        coeffs = [
            self.rng.uniform(-0.02, 0.02) for _ in range(8)
        ]
        return image.transform(
            (w, h), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR
        )
    
    # ── Quality Augmentations ──
    
    def _random_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """Random brightness and contrast adjustment."""
        brightness = self.rng.uniform(0.8, 1.2)
        contrast = self.rng.uniform(0.8, 1.3)
        
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return image
    
    def _random_noise(self, image: Image.Image) -> Image.Image:
        """Add random Gaussian noise."""
        arr = np.array(image, dtype=np.float32)
        noise_level = self.rng.uniform(5, 20)
        noise = np.random.normal(0, noise_level, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _random_blur(self, image: Image.Image) -> Image.Image:
        """Apply random Gaussian blur."""
        radius = self.rng.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        import io
        quality = self.rng.randint(30, 80)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    
    def _print_scan_simulation(self, image: Image.Image) -> Image.Image:
        """Simulate print-then-scan artifacts."""
        result = self._random_blur(image)
        result = self._random_noise(result)
        result = self._random_brightness_contrast(result)
        result = self._jpeg_compression(result)
        return result
    
    # ── Semantic Tampering (Synthetic Forgeries) ──
    
    def _copy_move_tamper(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Copy a region and paste it elsewhere in the document."""
        w, h = image.size
        region_w = self.rng.randint(w // 8, w // 4)
        region_h = self.rng.randint(h // 8, h // 4)
        
        # Source region
        src_x = self.rng.randint(0, w - region_w)
        src_y = self.rng.randint(0, h - region_h)
        region = image.crop((src_x, src_y, src_x + region_w, src_y + region_h))
        
        # Destination (far from source)
        dst_x = self.rng.randint(0, w - region_w)
        dst_y = self.rng.randint(0, h - region_h)
        
        image.paste(region, (dst_x, dst_y))
        return image, "copy_move"
    
    def _text_overlay_tamper(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Overlay synthetic text on the document."""
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        # Generate random text overlay
        texts = ["$99,999.00", "APPROVED", "VERIFIED", "0000", "XXXX"]
        text = self.rng.choice(texts)
        
        x = self.rng.randint(10, w - 100)
        y = self.rng.randint(10, h - 30)
        
        # Use default font (no external font dependency)
        draw.text((x, y), text, fill=(0, 0, 0))
        
        return image, "text_overlay"
    
    def _region_erase_tamper(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Erase a region (simulate white-out / redaction)."""
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        erase_w = self.rng.randint(w // 10, w // 5)
        erase_h = self.rng.randint(h // 20, h // 10)
        
        x = self.rng.randint(0, w - erase_w)
        y = self.rng.randint(0, h - erase_h)
        
        # Fill with near-white (simulating erasure)
        fill_color = (self.rng.randint(240, 255), self.rng.randint(240, 255), self.rng.randint(240, 255))
        draw.rectangle([x, y, x + erase_w, y + erase_h], fill=fill_color)
        
        return image, "region_erase"
    
    def _brightness_patch_tamper(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Create a brightness-inconsistent patch (simulates splicing)."""
        arr = np.array(image, dtype=np.float32)
        h, w = arr.shape[:2]
        
        patch_w = self.rng.randint(w // 6, w // 3)
        patch_h = self.rng.randint(h // 6, h // 3)
        
        x = self.rng.randint(0, w - patch_w)
        y = self.rng.randint(0, h - patch_h)
        
        # Apply different brightness to patch
        factor = self.rng.uniform(0.7, 1.4)
        arr[y:y+patch_h, x:x+patch_w] = np.clip(
            arr[y:y+patch_h, x:x+patch_w] * factor, 0, 255
        )
        
        return Image.fromarray(arr.astype(np.uint8)), "brightness_patch"
    
    def augment_dataset(self, input_dir: str, output_dir: str,
                        augmentations_per_image: int = 5,
                        generate_synthetic_tampers: bool = True):
        """Augment an entire dataset directory.
        
        Args:
            input_dir: Directory with subdirs 'original/' and 'forged/'.
            output_dir: Directory to save augmented images.
            augmentations_per_image: Number of augmented versions per image.
            generate_synthetic_tampers: Also generate synthetic tampers from clean images.
        """
        os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "forged"), exist_ok=True)
        
        # Process clean images
        original_dir = os.path.join(input_dir, "original")
        if os.path.exists(original_dir):
            for fname in os.listdir(original_dir):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img = Image.open(os.path.join(original_dir, fname)).convert("RGB")
                base_name = os.path.splitext(fname)[0]
                
                # Geometric/quality augmentations (keep label = clean)
                augmented = self.augment(img, label=0, num_augmentations=augmentations_per_image)
                for i, (aug_img, _) in enumerate(augmented):
                    aug_img.save(os.path.join(output_dir, "original", f"{base_name}_aug{i}.jpg"))
                
                # Synthetic tampers (label = forged)
                if generate_synthetic_tampers:
                    for j in range(augmentations_per_image):
                        tampered, tamper_type = self.synthesize_tamper(img.copy())
                        tampered.save(
                            os.path.join(output_dir, "forged", 
                                       f"{base_name}_synthetic_{tamper_type}_{j}.jpg")
                        )
        
        # Process forged images (augment only, keep label = forged)
        forged_dir = os.path.join(input_dir, "forged")
        if os.path.exists(forged_dir):
            for fname in os.listdir(forged_dir):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img = Image.open(os.path.join(forged_dir, fname)).convert("RGB")
                base_name = os.path.splitext(fname)[0]
                
                augmented = self.augment(img, label=1, num_augmentations=augmentations_per_image)
                for i, (aug_img, _) in enumerate(augmented):
                    aug_img.save(os.path.join(output_dir, "forged", f"{base_name}_aug{i}.jpg"))
        
        # Count results
        n_original = len(os.listdir(os.path.join(output_dir, "original")))
        n_forged = len(os.listdir(os.path.join(output_dir, "forged")))
        print(f"✅ Augmentation complete:")
        print(f"   Original (clean): {n_original} images")
        print(f"   Forged (tampered): {n_forged} images")
        print(f"   Total: {n_original + n_forged} images")


if __name__ == "__main__":
    augmentor = DocumentAugmentor()
    augmentor.augment_dataset(
        input_dir="images/training",
        output_dir="images/augmented",
        augmentations_per_image=5,
        generate_synthetic_tampers=True,
    )
