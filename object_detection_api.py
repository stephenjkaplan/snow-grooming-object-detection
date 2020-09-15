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

    :param bytes image_bytes:
    :return:
    :rtype:
    """
    my_transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)


def format_output(prediction, threshold):
    """

    :param prediction:
    :param threshold:
    :return:
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

    :param image_bytes:
    :param float threshold:
    :return:
    :rtype: list
    """
    tensor = transform_image(image_bytes=image_bytes)
    prediction = MODEL.forward(tensor)

    return format_output(prediction, threshold)


@app.route('/predict', methods=['POST'])
def predict():
    """

    :return:
    """
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        threshold = request.args.get('threshold', None)
        threshold = 0.5 if threshold is None else float(threshold)
        output = get_prediction(img_bytes, threshold)

        return jsonify(output)
