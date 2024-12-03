# Enhanced YOLO Model with Cross-Spatial-Channel Attention

## Introduction

In vehicle safety detection, identifying body markers on medium and large vehicles is crucial for ensuring safe road travel. This study proposes an enhanced Vehicle Body Markers YOLO (VBM-YOLO) model to address the information loss issues in previous YOLO series models. VBM-YOLO integrates the Cross-Spatial-Channel Attention (**CSCA**) mechanism, which leverages cross-dimensional information to establish global dependencies between spatial and channel dimensions, enriching feature representations. Additionally, we introduce the Multi-Scale Selective Feature Pyramid Network (**MSSFPN**), which employs a progressive fusion approach to mitigate feature loss and target layer information confusion. To further reduce gradient information loss during training, an Auxiliary Gradient Branch (**AGB**) is proposed, supervising gradient backpropagation without participating in inference. 

## The three improvements in our paper

#### **CSCA**

We have detailed the CSCA code in `ultralytics\nn\modules\CSCA.py` and added comments at key points to help readers better understand our code.

#### **MSSFPN**

We have used the MSSFPN structure in `ultralytics\cfg\models\VBM\VBM_YOLO.yaml` and added comments to indicate its location and overall architecture for the readers. For detailed modules of MSSFPN, please refer to `ultralytics\nn\modules\block.py`, where we have provided detailed explanations and comments at the corresponding locations of each module.

#### **AGB**

We have detailed the usage of AGB in the training process in `ultralytics\cfg\models\VBM\VBM_YOLO_AGB.yaml`. For the specifics of the modules in AGB, please refer to `ultralytics\nn\modules\block.py`, where we have provided comments on the key locations of each module.



## **Other details**

Referencing the experiments in the paper, first train using the `ultralytics/cfg/models/VBM/VBM_YOLO_AGB.yaml` configuration file, incorporating the AGB branch during training. After training is completed, the resulting weight file (.pt) should be **reparameterized** using `reparameterization.ipynb`. This process will reconstruct the trained parameters back into `ultralytics/cfg/models/VBM/VBM_YOLO`, producing a new weight file (.pt).



Just as mentioned in the paper, we utilize structural **reparameterization** in **C2f_Rep** to reduce GFLOPs. For more details, please refer to the specific implementation of **C2f_Rep** in `ultralytics\nn\modules\block.py` and the reparameterization method defined in `ultralytics\nn\tasks.py` under the function `def fuse`



## Get Start

### **Training & testing**

**dataset**

PASCAL VOC : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

D-Fire:[gaiasd/DFireDataset: D-Fire: an image data set for fire and smoke detection.](https://github.com/gaiasd/DFireDataset)

vehicle body markers dataset (our dataset) :https://pan.baidu.com/s/1hDMtZk-nAwgu7JG4taJWqQ  code:qpzy    or DOI: https://doi.org/10.5281/zenodo.14264265



**Train**

If you want to start training, first set up the environment according to `requirements.txt`. Then, specify the corresponding data path in `ultralytics\cfg\datasets\car.yaml`.Finally, execute the command `python train.py` to start training.

**Test**

The `test.py` script is used for performing inference on images and videos using the trained model.

**val**

The `val.py` script is used to evaluate the size of our model.
