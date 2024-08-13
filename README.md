# Label-studio-local-backend-YOLOv8
This repo contains the steps for initializing and using label studio's backend feature to automate annotations using YOLOv8 (detection and segmentation) locally without using docker

# Steps to use ML-backend locally on label-studio:
Installing the backend ML repo
-  ```git clone https://github.com/HumanSignal/label-studio-ml-backend.git```
-   ```cd label-studio-ml-backend/```
-   ```pip install -e .```

  
# Create an empty ML backend
-   ```label-studio-ml create my_ml_backend ```
	
# Implementing prediction logic
- copy attached model.py and _wsgi.py into their corresponding py files in my_ml_backend directory

# Running the ML-backend
-  ```pip install -r my_ml_backend```
-  ```label-studio-ml start my_ml_backend```


# Note:
 - You'll need to copy your LS-API key found in (Profile (in top right corner) --> Account and Settings --> Access token)
in LS API key variable at the top of the script 

- Attached below is model.py (for YOLOv8 segmentation) and model_det.py (for YOLOv8 detection), orginally from (https://github.com/seblful/label-studio-yolov8-backend) directory

- You'll need to change the model path to your model path (Found in model.py & model_det), and also change the labels in the labels list to your desired labels (make sure they match the labels on Label-Studio)


Helpful links:
- https://labelstud.io/guide/ml_create.html#Run-without-Docker (initialization steps)
- https://labelstud.io/guide/storage (Local storage integration)

 
