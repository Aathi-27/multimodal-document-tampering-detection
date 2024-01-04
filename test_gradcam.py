import tensorflow as tf
import numpy as np
from PIL import Image

from ela import convert_to_ela_image
from grad_cam import GradCAMExplainer

# Load model
model = tf.keras.models.load_model("model/1")

# Load & preprocess image
ela_img = convert_to_ela_image("images/predict/Paystub.jpg")
ela_arr = np.array(ela_img.resize((128, 128))) / 255.0
ela_arr = np.expand_dims(ela_arr, axis=0)

# Grad-CAM
explainer = GradCAMExplainer(model)
heatmap = explainer.make_heatmap(ela_arr)

print("Grad-CAM heatmap shape:", heatmap.shape)
print("Grad-CAM min/max:", heatmap.min(), heatmap.max())
