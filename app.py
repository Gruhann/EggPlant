import numpy as np
import tensorflow as tf
import cv2
import os
import base64
import uvicorn
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import io
from PIL import Image

app = FastAPI(title="Eggplant Disease Detection API",
              description="API for detecting health status of eggplants using YOLOv8 model",
              version="1.0.0")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model at startup to avoid loading it for each request
interpreter = None

# Class names for the model
class_names = ["Healthy", "Diseased"]  # Update these based on your model classes

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: List[int]

class PredictionResponse(BaseModel):
    detections: List[DetectionResult]
    image_b64: Optional[str] = None
    prediction_time_ms: float

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
async def predict_image(file: UploadFile = File(...), include_image: bool = Form(False)):
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
        orig_img = img.copy()

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
        
        # Start timer for performance measurement
        import time
        start_time = time.time()
        
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate prediction time
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Process results if in YOLOv8 format
        detections_list = []
        if len(output_data.shape) == 3 and output_data.shape[1] <= 85:
            # Transpose to make boxes along the batch dimension
            outputs = np.transpose(output_data[0], (1, 0))  # (8400, 7)
            
            # Get image dimensions
            img_height, img_width = orig_img.shape[:2]
            
            # Confidence threshold
            conf_threshold = 0.25
            
            # Process each prediction
            for i in range(len(outputs)):
                if outputs.shape[1] >= 6:
                    x, y, w, h = outputs[i][:4]
                    conf = outputs[i][4]
                    
                    if conf > conf_threshold:
                        class_id = int(np.argmax(outputs[i][5:]))
                        
                        # Convert to pixel coordinates
                        x1 = int((x - w/2) * img_width)
                        y1 = int((y - h/2) * img_height)
                        x2 = int((x + w/2) * img_width)
                        y2 = int((y + h/2) * img_height)
                        
                        # Ensure coordinates are within image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_width, x2), min(img_height, y2)
                        
                        # Create detection result
                        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                        detection = DetectionResult(
                            class_name=class_name,
                            confidence=float(conf),
                            box=[x1, y1, x2, y2]
                        )
                        
                        detections_list.append(detection)
                        
                        # Draw on image if we need to return it
                        if include_image:
                            color = (0, 255, 0) if "Healthy" in class_name else (0, 0, 255)
                            cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(orig_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        response = {
            "detections": [detection.dict() for detection in detections_list],
            "prediction_time_ms": prediction_time_ms
        }
        
        # Include the image in the response if requested
        if include_image:
            # Convert the image to base64 for JSON response
            _, buffer = cv2.imencode('.jpg', orig_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response["image_b64"] = img_base64
        
        return response
        
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