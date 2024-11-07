from ultralytics import YOLO 
from ultralytics import RTDETR
import torch


if __name__ == '__main__':
    # 加载模型  1.直接加载模型   2.加载模型结构 + 权重文件
    model = YOLO(r"ultralytics\cfg\models\v8\yolov8n.yaml")  # 不使用预训练权重训练
    # model.load(r'')
    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=r'ultralytics\cfg\datasets\car.yaml',
        epochs=300, 
        patience=50, 
        batch=8, 
        imgsz=640, 
        save=True, 
        save_period=-1, 
        cache=False,
        device='0', 
        workers=8,
        project='runs/train',  
        exist_ok=False,
        pretrained=False, 
        optimizer='SGD', 
        verbose=True, 
        seed=0, 
        deterministic=True, 
        single_cls=False,  
        rect=False, 
        cos_lr=False, 
        close_mosaic=10, 
        resume=False, 
        amp=True, 
        fraction=1.0, 
        profile=False, 
        freeze=None,
        # 分割
        overlap_mask=True, 
        mask_ratio=4,  
        # 分类
        dropout=0.0, 
        lr0=0.01, 
        lrf=0.01, 
        momentum=0.937, 
        weight_decay=0.0005, 
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1, 
        box=7.5, 
        cls=0.5, 
        dfl=1.5,
        pose=12.0, 
        kobj=1.0, 
        label_smoothing=0.0,  
        nbs=64, 
        hsv_h=0.015,  
        hsv_s=0.7,  
        hsv_v=0.4, 
        degrees=0.0, 
        translate=0.1, 
        scale=0.5,  
        shear=0.0,  
        perspective=0.0,  
        flipud=0.0, 
        fliplr=0.5,  
        mosaic=1.0, 
        mixup=0.0,  
        copy_paste=0.0,  
    )
