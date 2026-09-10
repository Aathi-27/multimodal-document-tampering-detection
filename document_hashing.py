"""Document integrity hashing module.

Generates perceptual hashes (pHash, dHash) and cryptographic hashes for documents.
Enables document comparison and tamper detection over time.

Usage:
    from document_hashing import DocumentHasher
    hasher = DocumentHasher()
    hashes = hasher.compute_hashes(image)
    is_similar = hasher.compare(hash1, hash2)
"""
import hashlib
import io
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


class DocumentHasher:
    """Compute and compare perceptual and cryptographic hashes for documents.
    
    Supports:
    - SHA-256: Exact binary match (any pixel change = different hash)
    - pHash (perceptual): Robust to minor quality changes, detects structural similarity
    - dHash (difference): Fast perceptual hash, detects layout/structure changes
    - aHash (average): Simple perceptual hash for quick comparisons
    """
    
    def __init__(self, hash_size: int = 16):
        """Initialize hasher.
        
        Args:
            hash_size: Size of perceptual hash grid (hash_size x hash_size).
                Larger = more sensitive to differences. Default 16.
        """
        self.hash_size = hash_size
    
    def compute_hashes(self, image: Image.Image) -> Dict[str, str]:
        """Compute all hash types for an image.
        
        Args:
            image: PIL Image to hash.
            
        Returns:
            Dict with keys: 'sha256', 'phash', 'dhash', 'ahash'.
        """
        return {
            "sha256": self.sha256_hash(image),
            "phash": self.perceptual_hash(image),
            "dhash": self.difference_hash(image),
            "ahash": self.average_hash(image),
        }
    
    def sha256_hash(self, image: Image.Image) -> str:
        """Compute SHA-256 hash of image bytes (exact binary match)."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()
    
    def perceptual_hash(self, image: Image.Image) -> str:
        """Compute perceptual hash using DCT (robust to compression artifacts).
        
        Resizes to hash_size×hash_size grayscale, applies DCT-like frequency
        analysis, and generates a binary hash from the median threshold.
        """
        img = image.convert("L").resize((self.hash_size * 4, self.hash_size * 4), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        
        # Simple DCT approximation via row/column means
        size = self.hash_size
        # Downsample to hash_size using block averaging
        block_h = pixels.shape[0] // size
        block_w = pixels.shape[1] // size
        reduced = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                reduced[i, j] = pixels[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w].mean()
        
        # Binary hash: above median = 1, below = 0
        median = np.median(reduced)
        bits = (reduced > median).flatten()
        return self._bits_to_hex(bits)
    
    def difference_hash(self, image: Image.Image) -> str:
        """Compute difference hash (dHash) — fast and effective.
        
        Compares adjacent pixel brightness: encodes gradient direction.
        Very fast, robust to scaling and minor quality changes.
        """
        img = image.convert("L").resize((self.hash_size + 1, self.hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        
        # Compare each pixel with its right neighbor
        bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
        return self._bits_to_hex(bits)
    
    def average_hash(self, image: Image.Image) -> str:
        """Compute average hash (aHash) — simplest perceptual hash.
        
        Resizes to hash_size×hash_size, computes mean, binary threshold.
        """
        img = image.convert("L").resize((self.hash_size, self.hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)
        mean = pixels.mean()
        bits = (pixels > mean).flatten()
        return self._bits_to_hex(bits)
    
    def compare(self, hash1: str, hash2: str) -> Tuple[int, float]:
        """Compare two hex hashes using Hamming distance.
        
        Args:
            hash1: First hash (hex string).
            hash2: Second hash (hex string).
            
        Returns:
            Tuple of (hamming_distance, similarity_percentage).
            - hamming_distance: Number of differing bits (lower = more similar).
            - similarity_percentage: 0.0 (completely different) to 1.0 (identical).
        """
        # Convert hex to binary
        try:
            bits1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bits2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        except ValueError:
            return -1, 0.0
        
        # Truncate to same length
        min_len = min(len(bits1), len(bits2))
        bits1 = bits1[:min_len]
        bits2 = bits2[:min_len]
        
        # Hamming distance
        distance = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
        similarity = 1.0 - (distance / min_len) if min_len > 0 else 0.0
        
        return distance, similarity
    
    def is_similar(self, hash1: str, hash2: str, threshold: float = 0.85) -> bool:
        """Check if two perceptual hashes are similar enough to be the same document.
        
        Args:
            hash1: First hash (hex string).
            hash2: Second hash (hex string).
            threshold: Minimum similarity (0.0-1.0) to consider "same document".
                Default 0.85 allows for minor quality differences.
        """
        _, similarity = self.compare(hash1, hash2)
        return similarity >= threshold
    
    @staticmethod
    def _bits_to_hex(bits: np.ndarray) -> str:
        """Convert boolean array to hex string."""
        bit_string = "".join("1" if b else "0" for b in bits)
        # Pad to multiple of 4 for clean hex conversion
        while len(bit_string) % 4 != 0:
            bit_string += "0"
        return hex(int(bit_string, 2))[2:]
