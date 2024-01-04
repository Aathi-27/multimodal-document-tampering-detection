"""Error Level Analysis (ELA) preprocessing module.

This module centralizes ELA image preprocessing to ensure training, inference (local and SageMaker),
and Streamlit app all use identical preprocessing logic. ELA detects image forgeries by analyzing
compression level differences within JPEG images.

Usage:
    from ela import convert_to_ela_image
    ela_image = convert_to_ela_image("path/to/image.jpg", quality=90)
"""
from PIL import Image, ImageChops, ImageEnhance


def convert_to_ela_image(path: str, quality: int = 90) -> Image.Image:
    """Convert a JPEG image to Error Level Analysis (ELA) representation.

    ELA works by re-encoding the image at a specified quality level, then comparing
    the original and re-encoded versions. Areas with identical compression are shown
    as black; areas that differ (potentially tampered) show as color gradients.

    Args:
        path: Absolute path to input JPEG image.
        quality: JPEG quality for re-encoding (0-100). Default 90 matches training data.

    Returns:
        PIL Image in RGB mode with ELA visualization. Brightness-enhanced for visibility.

    Notes:
        - Assumes input is JPEG format (lossy compression required for ELA).
        - Creates temporary file 'tempresaved.jpg' during processing.
        - All internal handles are closed; caller must close returned Image.
    """
    filename = path
    resaved_filename = 'tempresaved.jpg'
    im = Image.open(filename)
    bm = im.convert('RGB')
    im.close()
    im = bm
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    im.close()
    bm.close()
    resaved_im.close()
    del filename
    del resaved_filename
    del im
    del bm
    del resaved_im
    del extrema
    del max_diff
    del scale
    return ela_im
