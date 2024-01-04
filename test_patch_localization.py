import tensorflow as tf
import numpy as np
from PIL import Image

from ela import convert_to_ela_image
from patch_localization import PatchLocalizer

# Load model
model = tf.keras.models.load_model("model/1")

# Load & preprocess image
ela_img = convert_to_ela_image("images/predict/Paystub.jpg")
ela_arr = np.array(ela_img.resize((128, 128))) / 255.0

# Patch localization
localizer = PatchLocalizer(model)
heatmap = localizer.localize(ela_arr)

print("Patch localization heatmap shape:", heatmap.shape)
print("Patch localization min/max:", heatmap.min(), heatmap.max())
