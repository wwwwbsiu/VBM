import sys
sys.path.append("ultralytics")
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r".pt")  # load model weight
    model.predict(
        source=r"", # picture or file
    )
