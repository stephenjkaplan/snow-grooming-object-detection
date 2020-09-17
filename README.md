# Snow Grooming Object Detection

Check out the blog post (coming soon!) that provides a technical deep dive into this project.

#### Description

The objective of this project was to develop a proof of concept for autonomous snow grooming operations at ski resorts.
I did this by focusing on training a Faster R-CNN (neural network) object detection model in PyTorch to detect 
objects that might be of relevance to an autonomous vehicle moving through ski trails. I then applied that 
model to sample video footage simulating what a snow grooming vehicle might see while operating. 

![Model in Action 1](static/demo1.gif)

![Model in Action 2](static/demo2.gif)

This project was developed over a 3 week span as my final project for the [Metis](https://www.thisismetis.com/) data 
science program.

#### Data Sources
* [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)

#### File Contents
* `Object_Detection_Transfer_Learning_with_PyTorch.ipynb` should be used to train the neural network and use it to 
  perform object detection on videos. _It has only been used and tested with Google Colaboratory. It is strongly 
  recommended that you view/run it as a Colab Notebook._ You can do so by clicking the 'Open in Colab' button on 
  [GitHub](https://github.com/stephenjkaplan/snow-grooming-object-detection/blob/master/Object_Detection_Transfer_Learning_with_PyTorch.ipynb).
* `object_detection_api.py` contains a function and API endpoint that can be run on localhost for making boundary box predictions
   on any image you provide. However, this relies on the creation of a model using the notebook. No example model is 
   provided due to file size constraints of GitHub.
* `data_acquisition.py` contains a function for pulling images and annotated boundary boxes from the 
  [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html). 
* `dataset.py` contains a custom class inherited from the [PyTorch Dataset class](https://pytorch.org/docs/stable/data.html) 
   used for loading and processing data for training and validation.
* `utilities.py` contains a number of functions, many of which are for training and evaluating the object detection 
   model. Several of the functions are adapted from the 
   [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
* `torchvision_scripts/` contains files cloned from the [torchvision repository](https://github.com/pytorch/vision) that 
  aren't accessible via the normal install of torchvision.

#### Using Object Detection API

TI developed a prototype for an object detection API that can be run on localhost. You must first use the notebook 
in this repository to train and save a model. No pre-built model is provided. Assuming you have 
[Flask](https://flask.palletsprojects.com/en/1.1.x/) installed, first run the following command in your terminal:

`$ FLASK_ENV=development FLASK_APP=object_detection_api.py flask run`

Then, assuming you hae the [requests](https://requests.readthedocs.io/en/master/) library installed, make the following 
HTTP request:

```python
import requests

response = requests.post(
    url="http://localhost:5000/predict",
    files={"file": open('test_image.jpg', 'rb')}
)
```

Replacing the image file name with the image of interest. If you'd like to tune the predicted probability threshold 
for detection of an object class, you can pass it as a parameter:

```python
response = requests.post(
    url="http://localhost:5000/predict",
    params={'threshold': 0.25},
    files={"file": open('test_image.jpg', 'rb')}
)
```

You can also use the function `object_detection_api.get_prediction` directly (see the function for documentation.)


#### Dependencies

If you intend to run the notebook `Object_Detection_Transfer_Learning_with_PyTorch.ipynb`, installation
of dependencies is taken care of.

However if you intend to use the code in this repository "a la carte", you can install the packages via:

`pip install -r requirements.txt`