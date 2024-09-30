# AIMedVision-Backend
AIMedVision is a Flask-based REST API designed for medical image analysis and classification. This backend application supports image classification for two specific modalities: Colon Pathology and Chest X-Ray, utilizing pre-trained deep learning models to deliver accurate predictions.

# Features:
## REST API: 
Provides endpoints for seamless image classification.
## Model Selection: 
Users can choose between Colon Pathology and Chest X-Ray models for prediction.
## Image Upload: 
Allows users to upload images for classification from organized sample folders.
## Prediction Verification: 
Facilitates checking predictions against labeled filenames.

# Installation

## 01. Create a Conda environment and install dependencies:

conda create --name flask-app python=3.10
conda activate flask-app
pip install -r requirements.txt

## 02. Run the Flask application:

python app.py

## 03. Access the application at http://localhost:5000 in your web browser.

# Usage
01. Select the desired model (Colon Pathology or Chest X-Ray) for prediction.
02. Upload images from the organized_test_samples_Colon_Pathology folder for Colon Pathology model or the organized_test_samples_Chest_X-Ray folder for the Chest X-Ray model.
03. Verify predictions by checking the labels in the filenames of the uploaded images.

# API Endpoints
POST /classify: Upload an image for classification.
Parameters:
model: Model selection (colon_pathology or chest_xray).
image: Image file to classify.

# Folder Structure
01. app.py: Backend code for the Flask application.
02. index.html: Frontend HTML file.
03. organized_test_samples_Colon_Pathology: Folder containing test images for Colon Pathology.
04. organized_test_samples_Chest_X-Ray: Folder containing test images for Chest X-Ray.
05. requirements.txt: List of dependencies for Conda environment

