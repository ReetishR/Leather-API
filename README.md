# Leather Quality Classifier Web Application

This is a web application that uses a trained InceptionV3 model to classify leather quality. The model can classify leather images into 4 different categories.

## Features

- Modern, responsive web interface
- Drag and drop image upload
- Real-time image preview
- Classification results with confidence scores
- Support for common image formats (JPG, PNG)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd leather-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure you have the model file:
- The model file `inceptionNetV3_100eVAL_16b_v2_model.h5` should be in the root directory
- This file contains the trained InceptionV3 model for leather classification

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Simplest API Deployment (FastAPI)

For a minimal prediction service (no UI/login) use `simple_api.py`:

1. Install requirements (FastAPI/uvicorn already listed):
    ```bash
    pip install -r requirements.txt
    ```
2. Ensure the Inception `.h5` model is available and set the path (defaults to `inception.h5` in repo root):
    ```bash
    set MODEL_INCEPTION_PATH=inception.h5  # Windows PowerShell
    export MODEL_INCEPTION_PATH=inception.h5  # Unix shells
    ```
3. Start the API with uvicorn:
    ```bash
    uvicorn simple_api:app --host 0.0.0.0 --port 8000
    ```
4. Send predictions via HTTP:
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@path/to/leather.jpg"
    ```

This spins up a CORS-enabled FastAPI server that loads the model once and returns JSON predictions.

## Legacy implementation

The original Flask UI, templates, static assets, and alternate FastAPI stack were moved into `legacy/` for safekeeping:

- `legacy/flask_app/` – original `app.py`, templates, and static files
- `legacy/fullstack_api/` – earlier FastAPI service with database helpers
- `legacy/docs/` – previous Elastic Beanstalk runbook
- `legacy/models/` – unused/alternate `.h5` artifacts (e.g., AlexNet)

Only `simple_api.py` is required for the current proof-of-concept deployment.

## Usage

1. Click the upload area or drag and drop an image file
2. Preview the image
3. Click "Classify Leather" to get the prediction
4. View the classification result and confidence score

## Model Information

- Base Model: InceptionV3
- Input Size: 224x224 pixels
- Classes: 4 leather quality categories
- Training Accuracy: >97%
- Validation Accuracy: >97%

## Technical Details

- Frontend: HTML5, CSS3, JavaScript
- Backend: Flask (Python)
- Deep Learning: TensorFlow/Keras
- UI Framework: Bootstrap 5

## License

[Your License Information] 