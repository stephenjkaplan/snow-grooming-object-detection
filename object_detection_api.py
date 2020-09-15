"""
Contains function and API endpoint for making predictions on any image. You must first run the notebook
"Snow_Grooming_Object_Detection_with_PyTorch.ipynb" to generate the model. Otherwise, using this file will yield a
FileNotFoundError when it tries to load the model from its expected location (/models/final_model.pkl).
"""

import io
from PIL import Image
from itertools import compress
from flask import Flask, jsonify, request

import torch
import torchvision.transforms as transforms

# initialize global variables
app = Flask(__name__)
CLASS_INDEX = {1: 'tree', 2: 'person', 3: 'pole'}
MODEL = torch.load('models/final_model.pkl', map_location=torch.device('cpu'))
MODEL.eval()


def transform_image(image_bytes):
    """
    Performs necessary transformations on image data to prepare it for model. Converts it to a PyTorch tensor.

    :param bytes image_bytes: Bytes-type object containing image data. Using the native Python open() function will
                              yield this data type.
    :return: 1-dimensional PyTorch tensor.
    :rtype: torch.Tensor
    """
    my_transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)


def format_output(prediction, threshold):
    """
    Converts format of PyTorch object detection prediction to a list of dictionaries each containing information
    about a predicted bounding box.

    :param list prediction: Object detection prediction containing tensors for boundary boxes, labels, and scores.
    :param float threshold: Score threshold of predictions. Only predicted boundary boxes with corresponding scores
                            above this value will be returned.
    :return: List of dictionaries, each containing a score, boundary box coordinates, and label in text form.
    :rtype: list
    """
    # convert tensors to lists
    scores = prediction[0]['scores'].tolist()
    boxes = prediction[0]['boxes'].tolist()
    class_labels = prediction[0]['labels'].tolist()

    # get Boolean list for indices of score above threshold
    is_above_threshold = [True if s >= threshold else False for s in scores]

    # filter outputs
    output_scores = list(compress(scores, is_above_threshold))
    output_boxes = list(compress(boxes, is_above_threshold))
    output_class_labels = list(compress(class_labels, is_above_threshold))

    return [{
        'score': score,
        'box': box,
        'label': CLASS_INDEX[label]
    } for score, box, label in zip(output_scores, output_boxes, output_class_labels)]


def get_prediction(image_bytes, threshold=0.5):
    """
    Makes boundary box prediction on image data and puts results in a JSON-ish format.

    :param bytes image_bytes: Bytes-type object containing image data. Using the native Python open() function will
                              yield this data type.
    :param float threshold: Score threshold of predictions. Only predicted boundary boxes with corresponding scores
                            above this value will be returned.
    :return: List of dictionaries, each containing a score, boundary box coordinates, and label in text form.
    :rtype: list
    """
    tensor = transform_image(image_bytes=image_bytes)
    prediction = MODEL.forward(tensor)

    return format_output(prediction, threshold)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making boundary box predictions on an image. See README for sample code to access this endpoint.
    Only accepts POST requests.

    :return: List of dictionaries, each containing a score, boundary box coordinates, and label in text form.
    :rtype: list
    """
    if request.method == 'POST':
        # process data in request
        file = request.files['file']
        img_bytes = file.read()
        threshold = request.args.get('threshold', None)
        threshold = 0.5 if threshold is None else float(threshold)

        output = get_prediction(img_bytes, threshold)

        return jsonify(output)
