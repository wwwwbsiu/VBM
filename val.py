import sys
sys.path.append("ultralytics")
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'ultralytics\cfg\models\VBM\VBM_YOLO.yaml')  # build a new model from YAML
    model.info()