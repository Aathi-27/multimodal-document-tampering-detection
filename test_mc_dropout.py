import tensorflow as tf
import numpy as np
from ela import convert_to_ela_image
from mc_dropout import MCDropoutUncertainty

# Load model
model = tf.keras.models.load_model("model/1")

# Load & preprocess image
ela_img = convert_to_ela_image("images/predict/Paystub.jpg")
ela_arr = np.array(ela_img.resize((128, 128))) / 255.0
ela_arr = np.expand_dims(ela_arr, axis=0)

# MC Dropout
mc = MCDropoutUncertainty(model, num_samples=20, uncertainty_threshold=0.15)
result = mc.estimate(ela_arr)

print("MC Dropout result:")
print(result)
