"""Model optimization scripts for deployment.

Converts the trained EfficientNetB7 model to optimized formats:
- TensorFlow Lite (INT8/FP16 quantization)
- ONNX Runtime format
- TensorRT-ready SavedModel

Usage:
    python model_optimization.py --format tflite --output model_optimized/
    python model_optimization.py --format onnx --output model_optimized/
    python model_optimization.py --format fp16 --output model_optimized/
"""
import os
import sys
import argparse
import numpy as np


def optimize_tflite_int8(model_path: str, output_dir: str, 
                          representative_data_dir: str = None):
    """Convert Keras model to TFLite with INT8 quantization.
    
    INT8 quantization:
    - ~75% size reduction (260MB → ~65MB)
    - ~3-4× inference speedup on CPU
    - ~1-2% accuracy drop typical
    
    Args:
        model_path: Path to saved Keras model.
        output_dir: Directory to save optimized model.
        representative_data_dir: Directory with representative images for calibration.
    """
    import tensorflow as tf
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for calibration (improves accuracy)
    if representative_data_dir and os.path.exists(representative_data_dir):
        def representative_dataset():
            from PIL import Image
            image_files = [f for f in os.listdir(representative_data_dir) 
                          if f.endswith(('.jpg', '.png'))][:100]
            for fname in image_files:
                img = Image.open(os.path.join(representative_data_dir, fname)).convert("RGB")
                img = img.resize((128, 128))
                arr = np.array(img, dtype=np.float32) / 255.0
                yield [arr.reshape(1, 128, 128, 3)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    print("Converting to TFLite INT8...")
    tflite_model = converter.convert()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ Saved INT8 TFLite model to {output_path} ({size_mb:.1f} MB)")
    return output_path


def optimize_tflite_fp16(model_path: str, output_dir: str):
    """Convert Keras model to TFLite with FP16 quantization.
    
    FP16 quantization:
    - ~50% size reduction (260MB → ~130MB)
    - ~1.5-2× inference speedup
    - Minimal accuracy drop (<0.5%)
    
    Args:
        model_path: Path to saved Keras model.
        output_dir: Directory to save optimized model.
    """
    import tensorflow as tf
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    print("Converting to TFLite FP16...")
    tflite_model = converter.convert()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_fp16.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ Saved FP16 TFLite model to {output_path} ({size_mb:.1f} MB)")
    return output_path


def optimize_onnx(model_path: str, output_dir: str):
    """Convert Keras model to ONNX format for ONNX Runtime inference.
    
    ONNX Runtime:
    - ~2-3× inference speedup on CPU
    - Cross-platform deployment (C++, Java, C#, Python)
    - Hardware acceleration (GPU, NPU, TPU)
    
    Requires: pip install tf2onnx onnxruntime
    
    Args:
        model_path: Path to saved Keras model.
        output_dir: Directory to save ONNX model.
    """
    try:
        import tf2onnx
        import onnxruntime as ort
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Install with: pip install tf2onnx onnxruntime")
        return None
    
    import tensorflow as tf
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    
    # Convert to ONNX
    spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, 
                                                  opset=13, output_path=output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved ONNX model to {output_path} ({size_mb:.1f} MB)")
    
    # Verify with ONNX Runtime
    print("Verifying with ONNX Runtime...")
    session = ort.InferenceSession(output_path)
    dummy_input = np.random.randn(1, 128, 128, 3).astype(np.float32)
    outputs = session.run(None, {"input": dummy_input})
    print(f"✅ ONNX Runtime verification passed. Output shape: {outputs[0].shape}")
    
    return output_path


def optimize_saved_model_fp16(model_path: str, output_dir: str):
    """Save model with FP16 weights for TensorRT or GPU deployment.
    
    Args:
        model_path: Path to saved Keras model.
        output_dir: Directory to save FP16 model.
    """
    import tensorflow as tf
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert weights to FP16
    print("Converting weights to FP16...")
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            fp16_weights = [w.astype(np.float16) if w.dtype == np.float32 else w 
                           for w in weights]
            layer.set_weights(fp16_weights)
    
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    
    print(f"✅ Saved FP16 model to {output_dir}")
    return output_dir


def benchmark_model(model_path: str, num_runs: int = 100):
    """Benchmark model inference speed.
    
    Args:
        model_path: Path to model.
        num_runs: Number of inference runs for averaging.
        
    Returns:
        Dict with timing statistics.
    """
    import tensorflow as tf
    import time
    
    print(f"Benchmarking model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Warmup
    dummy = np.random.randn(1, 128, 128, 3).astype(np.float32)
    for _ in range(5):
        model(dummy, training=False)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model(dummy, training=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    stats = {
        "mean_ms": np.mean(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }
    
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"  Mean:   {stats['mean_ms']:.1f} ms")
    print(f"  Median: {stats['median_ms']:.1f} ms")
    print(f"  P95:    {stats['p95_ms']:.1f} ms")
    print(f"  P99:    {stats['p99_ms']:.1f} ms")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model optimization scripts")
    parser.add_argument("--format", choices=["tflite", "tflite_fp16", "onnx", "fp16", "benchmark"],
                       default="tflite", help="Optimization format")
    parser.add_argument("--model", default="model/1", help="Path to input model")
    parser.add_argument("--output", default="model_optimized", help="Output directory")
    parser.add_argument("--calibration_data", default=None, 
                       help="Directory with calibration images (for INT8)")
    
    args = parser.parse_args()
    
    if args.format == "tflite":
        optimize_tflite_int8(args.model, args.output, args.calibration_data)
    elif args.format == "tflite_fp16":
        optimize_tflite_fp16(args.model, args.output)
    elif args.format == "onnx":
        optimize_onnx(args.model, args.output)
    elif args.format == "fp16":
        optimize_saved_model_fp16(args.model, args.output)
    elif args.format == "benchmark":
        benchmark_model(args.model)
