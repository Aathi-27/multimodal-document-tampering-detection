"""Error Level Analysis (ELA) preprocessing module.

This module centralizes ELA image preprocessing to ensure training, inference (local and SageMaker),
and Streamlit app all use identical preprocessing logic. ELA detects image forgeries by analyzing
compression level differences within JPEG images.

Changelog:
    - v2: Added in-memory BytesIO processing (no disk I/O) for thread-safety and ~4x speedup.
    - v2: Added multi-quality ELA support for more robust tampering detection.
    - v2: Added PIL Image input support (no file path required).

Usage:
    from ela import convert_to_ela_image
    ela_image = convert_to_ela_image("path/to/image.jpg", quality=90)

    # Multi-quality ELA (new):
    from ela import convert_to_ela_image_multi
    ela_images = convert_to_ela_image_multi("path/to/image.jpg", qualities=[70, 80, 90, 95])
"""
import io
from PIL import Image, ImageChops, ImageEnhance


def convert_to_ela_image(path_or_image, quality: int = 90) -> Image.Image:
    """Convert a JPEG image to Error Level Analysis (ELA) representation.

    ELA works by re-encoding the image at a specified quality level, then comparing
    the original and re-encoded versions. Areas with identical compression are shown
    as black; areas that differ (potentially tampered) show as color gradients.

    OPTIMIZED: Uses in-memory BytesIO buffer instead of disk I/O.
    - ~4x faster than disk-based approach
    - Thread-safe (no shared temp files)
    - No filesystem cleanup needed

    Args:
        path_or_image: Either a file path (str) or a PIL Image to process.
        quality: JPEG quality for re-encoding (0-100). Default 90 matches training data.

    Returns:
        PIL Image in RGB mode with ELA visualization. Brightness-enhanced for visibility.

    Notes:
        - Assumes input is JPEG format (lossy compression required for ELA).
        - Uses in-memory buffer (no temp files created).
    """
    # Support both file path and PIL Image input
    if isinstance(path_or_image, str):
        im = Image.open(path_or_image).convert('RGB')
    else:
        im = path_or_image.convert('RGB')

    # In-memory re-encoding: avoids disk I/O entirely
    buffer = io.BytesIO()
    im.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved_im = Image.open(buffer)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    im.close()
    resaved_im.close()
    buffer.close()
    return ela_im


def convert_to_ela_image_multi(path_or_image, qualities: list = None) -> dict:
    """Multi-quality ELA: compute ELA at multiple compression levels.

    Different tampering techniques leave artifacts at different compression levels.
    Running ELA at multiple qualities provides a more robust detection signal.

    Args:
        path_or_image: Either a file path (str) or a PIL Image.
        qualities: List of JPEG quality levels. Default [70, 80, 90, 95].

    Returns:
        Dict mapping quality level to ELA PIL Image.
        Example: {70: <Image>, 80: <Image>, 90: <Image>, 95: <Image>}
    """
    if qualities is None:
        qualities = [70, 80, 90, 95]

    results = {}
    for q in qualities:
        results[q] = convert_to_ela_image(path_or_image, quality=q)
    return results
