import os
import random
import requests
import socket
from PIL import Image
from io import BytesIO

import numpy as np
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path


# hostname = socket.gethostname()
# LS_URL = socket.gethostbyname(hostname)
# print("Hostname: ", hostname)
# print("IP Address: ", ip_address)

LS_URL = "http://localhost:8080/"
LS_API_TOKEN = "2cd1da8014097f1bef4c7eb09c6c59f3cdc98052"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        self.labels = ['pothole', 'snow']
        # Load model
        self.model = YOLO(r"C:\Users\sdarwish\Datasets\Snow and potholes\Near_perfect\runs\segment\train3\weights\best.pt")

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN}

        # Getting URL and loading image
        image = Image.open(BytesIO(requests.get(
            full_url, headers=header).content))
        # Height and width of image
        original_width, original_height = image.size

        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        i = 0

        # Getting prediction using model
        results = self.model.predict(image)

        if results is None:
            print("No masks found in the result. Skipping this result.")
            return []

        # Getting mask segments, boxes from model prediction
        for result in results:
            if result.masks is None:
                print("No masks found for this result. Skipping.")
                continue

            if not hasattr(result, 'masks') or result.masks is None:
                print("Result does not have masks or masks is None.")
                continue

            if not hasattr(result.masks, 'xy') or result.masks.xy is None:
                print("Result masks does not have xy or xy is None.")
                continue

            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)): 
                if segm is None:
                    print(f"No segment found for box {i}. Skipping.")
                    continue
                # 2D array with poligon points
                polygon_points = segm / \
                    np.array([original_width, original_height]) * 100

                polygon_points = polygon_points.tolist()

                # Adding dict to prediction
                predictions.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "id": str(i),
                    "type": "polygonlabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_points,
                        "polygonlabels": [self.labels[int(box.cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()
        #score = score / (i + 1)
        print(f"Prediction Score is {score:.3f}.")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8m"
        }]

        return final_prediction

    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}
