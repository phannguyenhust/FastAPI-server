import tensorflow as tf

MODEL_PATH = "model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
try:
    interpreter.allocate_tensors()
    print("Model loaded and tensors allocated successfully.")
except RuntimeError as e:
    print(f"Failed to allocate tensors: {e}")

