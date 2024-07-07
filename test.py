from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = FastAPI()

MODEL_PATH = "mobilenetv2_model.h5"
DICT_PATH = "dict.txt"

# Load the model
model = load_model(MODEL_PATH)

# Load class names from the dictionary file
with open(DICT_PATH, 'r') as file:
    class_names = [line.strip().split(':')[1] for line in file.readlines()]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check if an image file was uploaded
    if file is None:
        return JSONResponse(content={"error": "No image received."})

    # Read the image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check if the image was successfully read
    if img is None:
        return JSONResponse(content={"error": "Cannot read image file."})

    # Resize the image to match the input shape of the model
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]

    return JSONResponse(content={"prediction": predicted_class_name, "confidence": float(np.max(predictions))})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
