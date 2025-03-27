import numpy as np
import tensorflow as tf
import cv2
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Eggplant Disease Detection API",
              description="API for detecting health status of eggplants using YOLOv8 model",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
interpreter = None

# Class names for the model
class_names = ["Healthy", "Diseased"]

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float

@app.on_event("startup")
async def startup_event():
    global interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path="Quality.tflite")
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Eggplant Disease Detection API is running. Use /predict endpoint to analyze images."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    global interpreter
    
    if interpreter is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please try again later."}
        )

    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image format. Please upload a valid image."}
            )

        # Get model input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Process image for model input
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # Resize image to model input size
        img_resized = cv2.resize(img, (input_width, input_height))
        img_normalized = img_resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(img_normalized, axis=0)
        
        # Set the input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        if len(output_data.shape) == 3 and output_data.shape[1] <= 85:
            # Transpose to make boxes along the batch dimension
            outputs = np.transpose(output_data[0], (1, 0))
            
            # Get the highest confidence prediction
            max_conf = 0
            predicted_class = None
            
            for i in range(len(outputs)):
                if outputs.shape[1] >= 6:
                    conf = outputs[i][4]
                    if conf > max_conf:
                        class_id = int(np.argmax(outputs[i][5:]))
                        max_conf = conf
                        predicted_class = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            if predicted_class is not None:
                return {
                    "class_name": predicted_class,
                    "confidence": float(max_conf)
                }
        
        return JSONResponse(
            status_code=400,
            content={"error": "No valid predictions found in the image."}
        )
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# For local development
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 