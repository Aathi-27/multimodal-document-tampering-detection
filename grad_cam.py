"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module.

This module generates visual explanations of CNN predictions by highlighting the regions
that most strongly influenced the model's classification decision. Uses gradients from the
final convolutional layer to create class activation heatmaps.

Key insight: Grad-CAM does NOT modify the model weights, architecture, or inference flow.
It is a post-hoc visualization technique applied to existing predictions.

Usage:
    from grad_cam import GradCAMExplainer
    explainer = GradCAMExplainer(model)
    heatmap = explainer.make_heatmap(img_array)
    overlay = explainer.overlay_heatmap(original_image, heatmap)
    overlay.save("visualization.png")
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


class GradCAMExplainer:
    """Grad-CAM explainability for a Keras CNN model.
    
    Generates class activation heatmaps using gradients from the final convolutional layer.
    Does not alter model weights or predictions.
    """
    
    def __init__(self, model: tf.keras.Model):
        """Initialize Grad-CAM explainer for a model.
        
        Args:
            model: Trained Keras/TensorFlow CNN model.
            
        Raises:
            ValueError: If no convolutional layer found in the model.
        """
        self.model = model
        self.last_conv_layer_name = self._find_last_conv_layer()
        
    def _find_last_conv_layer(self) -> str:
        """Locate the final convolutional layer for gradient computation.
        
        Iterates through model layers in reverse order to find the last Conv2D.
        
        Returns:
            Name of the last convolutional layer.
            
        Raises:
            ValueError: If no convolutional layer found.
        """
        for layer in reversed(self.model.layers):
            try:
                # Conv layers have 4D output: (batch, height, width, channels)
                if len(layer.output_shape) == 4:
                    return layer.name
            except Exception:
                continue
        raise ValueError("No convolutional layer found in model for Grad-CAM.")
    
    def make_heatmap(self, img_array: np.ndarray) -> np.ndarray:
        """Generate Grad-CAM heatmap for an input image.
        
        Computes gradients of the predicted class with respect to the final conv layer's
        output, then multiplies activation maps by these gradients to highlight influential regions.
        
        Args:
            img_array: Input array with shape (1, height, width, 3) and values in [0, 1].
            
        Returns:
            Heatmap as (height, width) numpy array with values in [0, 1].
        """
        # Build auxiliary model to access final conv layer outputs and model predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]  # type: ignore[union-attr]
        )
        
        # Compute gradients of predicted class w.r.t. final conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Get the class with highest prediction (assuming 2-class: [original, tampered])
            top_class = tf.argmax(predictions[0])
            top_class_channel = predictions[:, top_class]
        
        grads = tape.gradient(top_class_channel, conv_outputs)
        
        # Weight activations by gradients: average gradient across spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Linear combination: activation map * gradient weights
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU: only positive gradients indicate influence
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """Overlay Grad-CAM heatmap on original image.
        
        Resizes heatmap to match image dimensions, applies a JET colormap, and blends
        with the original image using alpha transparency.
        
        Args:
            original_image: PIL Image (RGB) to overlay heatmap on.
            heatmap: (height, width) array with values in [0, 1].
            alpha: Transparency factor for heatmap (0=original only, 1=heatmap only).
            
        Returns:
            PIL Image with heatmap overlaid (RGB mode).
        """
        base = original_image.convert("RGB")
        
        # Resize heatmap to match image dimensions
        heat_resized = cv2.resize(heatmap, base.size)
        
        # Convert to uint8 for colormap: applyColorMap requires single-channel uint8 input
        # Clip to [0,1] first, then scale to 0-255 range, then cast to uint8
        heat_clipped = np.clip(heat_resized, 0.0, 1.0).astype(np.float64)  # type: ignore[arg-type]
        heat_uint8 = (heat_clipped * 255.0).astype(np.uint8)
        
        # Apply JET colormap (blue=low, red=high)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # type: ignore[arg-type]
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        
        # Blend original and heatmap
        # Type: ignore because np.ndarray multiplication with float is valid for OpenCV blending
        overlay = np.array(base) * (1 - alpha) + heat_color * alpha  # type: ignore[operator]
        
        return Image.fromarray(np.uint8(overlay))
