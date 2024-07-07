import tensorflow as tf

# Đường dẫn tới mô hình .tflite
MODEL_PATH = "model.tflite"

# Tạo interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Lấy thông tin chi tiết về các tensor đầu vào và đầu ra
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Thong tin tensor dau vao:")
for detail in input_details:
    print(detail)

print("\nThong tin tensor dau ra:")
for detail in output_details:
    print(detail)

# Hiển thị thông tin về các lớp trong mô hình
print("\nThong tin ve cac lop trong mo hinh:")
for i in range(len(interpreter.get_tensor_details())):
    layer = interpreter.get_tensor_details()[i]
    print(f"Layer {i}: {layer['name']} - shape: {layer['shape']}, dtype: {layer['dtype']}")