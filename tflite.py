from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
import os

app = FastAPI()

MODEL_PATH = "mobilenet.tflite"
DICT_PATH = "dict.txt"

with open(DICT_PATH, 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Kiểm tra xem ảnh có được nhận không
    if file is None:
        return JSONResponse(content={"error": "No image received."})

    # Đọc ảnh từ file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Kiểm tra xem tệp ảnh có được đọc không
    if img is None:
        return JSONResponse(content={"error": "Cannot read image file."})

    # Chuyển đổi kích thước cho phù hợp với mô hình
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    img_resized = cv2.resize(img, (input_width, input_height))

    # Chuyển đổi ảnh sang dạng UINT8 nếu mô hình yêu cầu
    if input_details[0]['dtype'] == np.uint8:
        input_data = np.expand_dims(img_resized.astype(np.uint8), axis=0)
    else:
        input_data = np.expand_dims(img_resized.astype(np.float32), axis=0) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data) + 1  # Tăng chỉ số lên 1
    predicted_class_name = class_names[predicted_class - 1]  # Lấy tên lớp từ danh sách

    return JSONResponse(content={"prediction": predicted_class_name})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
