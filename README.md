# Enhanced YOLO Model with Cross-Spatial-Channel Attention

## Introduction
In vehicle safety detection, identifying body markers on medium and large vehicles is crucial for ensur-
ing safe road travel. This study proposes an enhanced Vehicle Body Markers YOLO (VBM-YOLO)
model to address the information loss issues in previous YOLO series models. VBM-YOLO inte-
grates the Cross-Spatial-Channel Attention (CSCA) mechanism, which leverages cross-dimensional
information to establish global dependencies between spatial and channel dimensions, enriching fea-
ture representations. Additionally, we introduce the Multi-Scale Selective Feature Pyramid Network
(MSSFPN), which employs a progressive fusion approach to mitigate feature loss and target layer
information confusion. To further reduce gradient information loss during training, an Auxiliary
Gradient Branch (AGB) is proposed, supervising gradient backpropagation without participating
in inference. Experimental results demonstrate that VBM-YOLO improves mean Average Precision
(mAP) by 2.3% and 4.3% at IoU thresholds of 0.5 and 0.5:0.95, respectively, compared to YOLOv8s on
the vehicle body markers dataset. VBM-YOLO also achieves a better balance between accuracy and
computational resources than other mainstream models, exhibiting good generalization performance
on public datasets like PASCAL VOC and D-Fire.

The recognizability and protectiveness  of vehicles themselves are crucial to traffic safety. Particularly for medium and large vehicles, their safety while on the road is intimately tied to the presence of their vehicle body markers. The primary vehicle body markers include rear reflective signs, side reflective signs, rear signs, rear guards, side guards. These markers significantly enhance the visibility and protective capabilities of vehicles under varying visibility conditions and road situations. Specifically, in environments characterized by low visibility at night or due to restricted lines of sight, reflective signs notably enhance the recognizability of vehicles, effectively reducing the risk of collision. For large transport vehicles, the conspicuous design of rear signs effectively prompts following vehicles to maintain a necessary, safe distance, thus preventing rear-end collisions. Furthermore, guard boards can mitigate collision risks and prevent small vehicles from being entrapped under the chassis of medium and large vehicles in the event of an accident. Given the essential role of triangle signs in all vehicles, they are considered an extension of the vehicle body safety markers. In the case of vehicle malfunction requiring parking, a triangle warning sign serves as a temporary safety warning facility, safeguarding the vehicle's and its occupants' safety. Consequently, before permitting normal vehicle operation, relevant authorities should inspect these vehicle body markers to ensure road safety compliance. This process can be effectively achieved using object detection technology based on computer vision.


## reproduce
Referencing the experiments in the paper, first train using the `ultralytics/cfg/models/VBM/VBM_YOLO_AGB.yaml` configuration file, incorporating the AGB branch during training. After training is completed, the resulting weight file (.pt) should be reparameterized using `reparameterization.ipynb`. This process will reconstruct the trained parameters back into `ultralytics/cfg/models/VBM/VBM_YOLO`**, producing a new weight file (.pt).

## Get Start
### **Training & testing**

**1.Train   ()**

`python train.py`

**2.Test**

`python test.py`

### Dataset

link：https://pan.baidu.com/s/1hDMtZk-nAwgu7JG4taJWqQ 

Please contact  bw23288@gmail.com to obtain the extraction code.
