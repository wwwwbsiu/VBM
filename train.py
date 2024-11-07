from ultralytics import YOLO 
from ultralytics import RTDETR
import torch


if __name__ == '__main__':
    model = YOLO(r"ultralytics\cfg\models\v8\yolov8n.yaml")  # load model cfg
    # model.load(r'') # pre-trained weights
    model.train(
        data=r'ultralytics\cfg\car.yaml', # load dataset cfg
    )
