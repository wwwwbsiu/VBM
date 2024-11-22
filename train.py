from ultralytics import YOLO 


if __name__ == '__main__':
    model = YOLO(r"ultralytics\cfg\models\VBM\VBM_YOLO_AGB.yaml")  # load model cfg
    #model = YOLO(r"ultralytics\cfg\models\VBM\VBM_YOLO.yaml")
    # model.load(r'') # pre-trained weights
    model.train(
        data=r'ultralytics\cfg\car.yaml', # load dataset cfg
        epochs=200,
        momentum=0.937,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        imgsz=640,
    )
