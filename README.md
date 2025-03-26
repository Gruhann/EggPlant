# Eggplant Disease Detection API

This is a web service that uses a YOLOv8 model to detect diseases in eggplants. The service provides an API endpoint that accepts an image and returns detection results.

## Files in this project

- `app.py` - The FastAPI application that serves the model
- `Quality.tflite` - The TensorFlow Lite model for eggplant disease detection 
- `requirements.txt` - Dependencies required to run the service
- `render.yaml` - Configuration for deployment on Render

## Deployment on Render

1. Sign up for a Render account at https://render.com
2. Create a new Web Service
3. Connect your GitHub repository containing this code
4. Select "Python" as the environment
5. Set the Build Command to: `pip install -r requirements.txt`
6. Set the Start Command to: `uvicorn app:app --host 0.0.0.0 --port $PORT`
7. Click "Create Web Service"

## API Usage

Once deployed, the API provides the following endpoints:

### 1. Health Check

```
GET /
```

Returns a simple message to confirm the API is running.

### 2. Predict Image

```
POST /predict
```

Parameters:
- `file`: The image file to analyze (multipart form data)
- `include_image`: Boolean to request the annotated image in the response (default: false)

Example response:
```json
{
  "detections": [
    {
      "class_name": "Healthy",
      "confidence": 0.85,
      "box": [10, 20, 100, 200]
    },
    {
      "class_name": "Diseased",
      "confidence": 0.75,
      "box": [150, 160, 300, 400]
    }
  ],
  "image_b64": "base64_encoded_image_string_if_requested",
  "prediction_time_ms": 123.45
}
```

## Local Development

To run the service locally:

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have the `Quality.tflite` model file in the project directory

3. Run the service:
   ```
   uvicorn app:app --reload
   ```

4. Access the API documentation at http://localhost:8000/docs

## Client Example (Python)

```python
import requests
from PIL import Image
import io
import base64

# API endpoint
url = "https://your-render-service-url.onrender.com/predict"

# Load and prepare the image
image_path = "eggplant.jpg"
with open(image_path, "rb") as image_file:
    files = {"file": (image_path, image_file, "image/jpeg")}
    data = {"include_image": "true"}
    
    # Send the request
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Print detection results
        for detection in result["detections"]:
            print(f"Detected: {detection['class_name']} with confidence {detection['confidence']}")
        
        # If you requested the annotated image
        if "image_b64" in result:
            image_data = base64.b64decode(result["image_b64"])
            image = Image.open(io.BytesIO(image_data))
            image.save("result.jpg")
            print("Annotated image saved as result.jpg")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
``` 