from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Đường dẫn đến mô hình và tệp nhãn
MODEL_PATH = "inceptionv3_model.h5"
DICT_PATH = "dict.txt"

# Tải mô hình và đọc nhãn lớp
model = load_model(MODEL_PATH)
with open(DICT_PATH, 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

def predict_image(image):
    # Xử lý ảnh
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Ghi log
    log_info = {
        "shape": img_array.shape,
        "dtype": img_array.dtype,
        "min_value": np.min(img_array),
        "max_value": np.max(img_array)
    }
    print("Input image array info:", log_info)

    # Dự đoán
    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    return {"prediction": predicted_class_name}



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = Image.open(io.BytesIO(nparr)).convert('RGB')
    img = img.resize((224, 224))  # Đảm bảo kích thước ảnh phù hợp với mô hình

    # Dự đoán
    result = predict_image(img)
    
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
