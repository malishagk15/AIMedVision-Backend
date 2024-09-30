from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the paths to the models
model_paths = {
    'model1': 'model2.h5',
    'model2': 'model1_pneumoniamnist.h5'
}

# Load the models
models = {}
for key, path in model_paths.items():
    models[key] = load_model(path)

# Define label mappings for each model
label_mappings = {
    'model1': {
        0: 'ADI',
        1: 'BACK',
        2: 'DEB',
        3: 'LYM',
        4: 'MUC',
        5: 'MUS',
        6: 'NORM',
        7: 'STR',
        8: 'TUM'
    },
    'model2': {
        0: 'Normal',
        1: 'BACTERIA or VIRUS'
    }
}

def process_image(image, model_name):
    if model_name == 'model1':
        # Resize the image to match the input shape of model1 (28x28)
        resized_image = cv2.resize(image, (28, 28))

        # Convert the image to float32 and normalize pixel values
        resized_image = resized_image.astype('float32') / 255

        # Make sure the image has 3 channels (for RGB)
        if resized_image.shape[-1] != 3:
            raise ValueError("Image does not have 3 channels (RGB)")

        # Reshape the image to match the input shape of model1 (1, 28, 28, 3)
        processed_image = np.reshape(resized_image, (1, 28, 28, 3))

    elif model_name == 'model2':
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to match the input shape of model2 (28x28)
        resized_image = cv2.resize(grayscale_image, (28, 28))

        # Convert the image to float32 and normalize pixel values
        resized_image = resized_image.astype('float32') / 255

        # Reshape the image to match the input shape of model2 (1, 28, 28, 1)
        processed_image = np.reshape(resized_image, (1, 28, 28, 1))

    else:
        raise ValueError("Invalid model name")

    return processed_image

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file and model part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'model' not in request.form:
        return jsonify({'error': 'No model selected'})

    file = request.files['file']
    selected_model = request.form['model']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image using OpenCV
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Process the image
        processed_image = process_image(image, selected_model)

        # Make predictions
        predictions = models[selected_model].predict(processed_image)

        # Get the predicted class (index with highest probability)
        predicted_class = int(np.argmax(predictions))

        # Get the corresponding label for the predicted class
        predicted_label = label_mappings[selected_model].get(predicted_class, 'Unknown')

        # Return the predicted class and label as JSON response
        return jsonify({'predicted_class': predicted_class, 'predicted_label': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)