import requests
import json

resp1 = requests.post(
    url="http://localhost:5000/predict",
    files={"file": open('test_image.jpeg', 'rb')}
)

resp2 = requests.post(
    url="http://localhost:5000/predict",
    params={'threshold': 0.25},
    files={"file": open('test_image.jpeg', 'rb')}
)

results1 = json.loads(resp1.content)
results2 = json.loads(resp2.content)
