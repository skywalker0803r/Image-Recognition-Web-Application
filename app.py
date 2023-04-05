# Import the required libraries
import os
from flask import Flask, request, render_template
import torch
import json
import cv2
import numpy as np

def load_model():
    # Load the model
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model.eval()
    return model

def preprocess_image(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (224,224))
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    img = img.transpose((0, 3, 1, 2))
    return img

def predict_image(model, image):
    # Disable autograd to improve performance
    with torch.no_grad():
        # Make a prediction using the model
        output = model(torch.from_numpy(image))

        # Convert the output to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Load the ImageNet class labels
        with open('imagenet_classes.json', 'r') as f:
            classes = json.load(f)

        # Get the index and class name of the highest probability
        index = torch.argmax(probabilities).item()
        class_name = classes[str(f"{int(index)}")]

        # Get the probability of the highest class
        probability = probabilities[index].item()

    return class_name, probability

# Create a Flask instance
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the image upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the file from the request
    file = request.files['file']

    # Save the file to disk
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Call the image recognition function and return the results
    result = recognize_image(filepath)
    result['filename'] = filename
    return render_template('result.html', result=result)

# Define the function for image recognition
def recognize_image(filepath):
    # Load the model
    # Here you would need to replace the model initialization with your own PyTorch model and the corresponding pre-processing code.
    # I assume you have already installed the PyTorch library and your model and pre-processing code are stored in a separate file.
    model = load_model()

    # Pre-process the image
    # Here you would need to add the pre-processing code for your specific model.
    # I assume the pre-processing code returns a tensor of the image data in the correct format for your model.
    image = preprocess_image(filepath)

    # Run the image through the model and get the top prediction
    # Here you would need to add the prediction code for your specific model.
    # I assume the prediction code returns a tuple with the top prediction label and confidence score.
    label, confidence = predict_image(model, image)

    # Return the result
    return {'label': label, 'confidence': confidence}

# Run the app
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

