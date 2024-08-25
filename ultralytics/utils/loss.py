# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

#-----------------------------------------------------------------------------------------#
#  VFL损失   
#  VFL = [(a*p^g)(1-label)+q*label]*BCE(p,q)         (a:alpha  p:pred_score.sigmoid()  g:gamma  q:gt_score)
#-----------------------------------------------------------------------------------------#
class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss





class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(
        self,
    ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()



#---------------------------------------------------------------------------------------#
#  回归损失： CIOU +  DFL
#---------------------------------------------------------------------------------------#

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    #----------------------------------------------------------------------------------------------------#
    #  pred_distri(pred_dist) ：torch.Size([16, 8400, 64]) 
    #                           预测框位置值(离散)

    #  pred_bboxes            : torch.Size([16, 8400, 4])
    #                           由pred_distri进行离散变化变为xyxy(不是原图大小)

    #  anchor_points          : torch.Size([8400, 2])
    #                           生成每个特征图的中心点坐标(x,y)(不是原图大小)

    #  target_bboxes          : torch.Size([16, 8400, 4])
    #                           每个预测框对应 gt 的xyxy

    #  target_scores          : torch.Size([16, 8400, 80])
    #                           将范围限制在正样本上 通过 one-hot编码后又用软标签 来标记预测框的类

    #  target_scores_sum      : target_scores.sum()
    #                           

    #  fg_mask                : torch.Size([16, 8400])  (布尔类型)
    #                           fg_mask代表的是有哪些锚点也就是预测框为1  也就是可以得到哪些预测框匹配到了gt（即哪些预测框为正样本）
    #-------------------------------------------------------------------------------------------------------#
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  #  正样本与gt求iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            #---------------------------------------------------------------------#
            # target_ltrb                                     ：torch.Size([16, 8400, 4])   
            #                                                 计算anchor point 到gt左上右下的距离

            # pred_dist.shape                                 ：torch.Size([16, 8400, 64])   
            # pred_dist[fg_mask].shape                        ：torch.Size([1958, 64])
            # pred_dist[fg_mask].view(-1, self.reg_max + 1)   ：torch.Size([6024, 16]) (正样本的位置离散值)

            # target_ltrb[fg_mask]                            ：torch.Size([1958, 4])  
            #                                                   anchor point 到 gt 左上右下的距离 (正样本与正样本对应的gt框)
            #---------------------------------------------------------------------#
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)   
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight  
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    #-------------------------------------------------------------#
    #  pred_dist：torch.Size([1184, 16]) 
    #             正样本的位置离散值    
     
    #  target：torch.Size([296, 4])  
    #          anchor point 到 gt 左上右下的距离 (正样本与正样本对应的gt框)
    #-------------------------------------------------------------#
    def _df_loss(pred_dist, target):  
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)







class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()




#----------------------------------------------------------------------------------------------------------------------#
#  损失计算
#  1.匹配正负样本（TaskAlignedAssigner） 如果grid_cell 的中点在gt内部，那么就由这些grid_cell产生的预测框与gt  
#    算 align_metric=bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)     
#    bbox_scores:选择预测框与gt相同类的预测概率  eg:预测框预测有80个类 它所在gt为a类，那么bbox_scores就为80个预测类中a的概率
#    overlaps.pow：为预测框与所在gt 的iou
#    算出每个预测框的 align_metric后 取align_metric分数前topk(默认10)的预测框，作为当前框的正样本，如果预测框匹配多个gt,那么哪个
#    预测框与哪个gt的iou高，此预测框就分配给哪个gt
#
#
#  2.计算损失
#    cls_loss(分类损失)：
#                      1.正负样本与gt框进行 BCE 计算   (正负样本都要参与)
                         #  self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
#                        #  pred_scores     ：torch.Size([16, 8400, 80])  预测框的类别预测信息  
                         #  target_scores   ：torch.Size([16, 8400, 80])  每个预测框 对应gt 类别信息  eg：[0,0,0,...,0.7,0,0](80)   [0,0,0,...,0,0,0](80) (负样本概率都为0)
#
#
#    reg_loss(回归损失)：
#                      1.正样本与其对应gt 进行 iou计算
#                      2.正样本与其对应gt 进行 dfl_loss计算
# 
#    
#


#  补充：  anchor_free  思想 ！！！！！！！！！！！！！！！！！！！  
#    Reg_max：需要额外指出的是，DFL有一个隐藏的超参数——Reg_max，这个参数代表了“输出特征图中，ltrb预测的最大范围 默认值为16，
#             pred_distri：torch.Size([16, 8400, 64]) 回归     64：4*16    4： 代表 ltrb(以grid_cell 为中心点 ltrb 为距中心点 左上右下的距离)
#             通过中心点和 ltrb来计算出 xyxy(对角坐标) 或者 xywh(中心点长宽)
#    
#    
#   
#   
#----------------------------------------------------------------------------------------------------------------------#
class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
  
        #--------------------VFL启用1------------------------#
        # self.varifocal_loss = VarifocalLoss()
        #--------------------END----------------------------#

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
   
        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out


#-------------------------------------------------------------------------------------------------#
#  anchor_free  思想 ！！！！！！！！！！！！！！！！！！！
    
#  pred_dist : torch.Size([16, 8400, 4])  4：ltrb 为以grid_cell的中点为中心点 ，ltrb 为中心点左上右下的距离（ltrb可以不一定一样）
#  dist2bbox : 通过中心点和 ltrb来计算出 xyxy(对角坐标) 或者 xywh(中心点长宽)
#-------------------------------------------------------------------------------------------------#

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))   # torch.Size([16, 8400, 4])
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        #--------------------------------------------------------#
        #  pred_scores：torch.Size([16, 8400, 80])  类
        #  pred_distri：torch.Size([16, 8400, 64])  坐标
        #--------------------------------------------------------#
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        #--------------------------------------------------------------------#
        # 原始图片大小(640x640)
        # anchor_points, stride_tensor：生成每个特征图的中心点坐标(x,y) 和 每个中心点坐标的步长
        #--------------------------------------------------------------------#
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w) 
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 生成每个特征图的中心点坐标(x,y) 和 每个中心点坐标的步长

        # Targets    
        #-----------------------------------------------------------------#
        # targets:shape[na, 6] 这里的na表示的是 16张图片 mosaic后的总的label数量, 6表示的是(batch_idx, cls, xywh)
        #-----------------------------------------------------------------#
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)   

        #------------------------------------------------------------------#
        #  targets:shape[16,max_label, 5]   
        #               max_label          ： 表示 单张图片中, label数量最多的个数max_label
        #               5                  :  表示 (cls,x,y,x,y) 且还原回原图

        #  self.preprocess                 :  将每张图片的label信息放入 size为（max_label, 5),不足的地方用0填充 
        #------------------------------------------------------------------#
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        #-------------------------------------------------------------#
        #  gt_labels：torch.Size([16, max_label, 1])
        #  gt_bboxes：torch.Size([16, max_label, 4])
        
        #  mask_gt：torch.Size([16, max_label])
        #  mask_gt 这个是用来判断是否有gt的, 先将xyxy的所有值求和, gt_(0)是大于0的置1, 否则置0
        #-------------------------------------------------------------#
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  

        # Pboxes  
        #---------------------------------------------#
        #  anchor_points   ：torch.Size([8400, 2])
        #  pred_distri     ：torch.Size([16, 8400, 64]) 
        #  pred_bboxes     ：torch.Size([16, 8400, 4]) xyxy
        #  作用            ：将pred_distri解码为边界框坐标 
        #---------------------------------------------#
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        #-------------------------------------------------------------------------------------------------------------------#
        #  pred_scores.detach().sigmoid()              ：torch.Size([16, 8400, 80])     预测框的类别预测信息
        #  (pred_bboxes.detach() * stride_tensor)      ：torch.Size([16, 8400, 4])      将预测的xyxy还原为真实图像的xyxy
        #  anchor_points * stride_tensor               ：torch.Size([8400, 2])          将预测框的中心点还原为真实图像
        #  gt_labels                                   ：torch.Size([16, max_label, 1]) 标签框的类别信息
        #  gt_bboxes                                   : torch.Size([16, max_label, 4]) 标签框的xyxy信息
        #  mask_gt                                     : torch.Size([16, max_label])    这个是用来判断是否有标签框
        #-------------------------------------------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------------------#
        #  作用               ：匹配正样本 通过 align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        #                       即将grid_cell中心点在gt框内部生成的预测框 与 gt框进行类别分数计算（bbox_scores.pow(self.alpha)）和 iou计算（overlaps.pow(self.beta)）
        #                       选则 align_metric 分数在前topk（10） 个预测框 作为正样本 ，对于一个预测框匹配多个gt框，选择与gt框更少的iou进行匹配
        #  返回     
        #  target_labels      ：torch.Size([16, 8400])
        #                       每个预测框的对应 gt 的类别

        #  target_bboxes      ：torch.Size([16, 8400, 4])
        #                       每个预测框对应 gt 的xyxy

        #  target_scores      ：torch.Size([16, 8400, 80])
        #                       将范围限制在正样本上 通过 one-hot编码 来标记预测框的类

        #  fg_mask.bool()     ：torch.Size([16, 8400])
        #                       fg_mask代表的是有哪些锚点也就是预测框为1  也就是可以得到哪些预测框匹配到了gt（即哪些预测框为正样本）

        #  target_gt_idx      ：torch.Size([16, 8400])
        #                       找到每一个预测框对应的gt框的索引

        #--------------------------------------------------------------------------------------------#
        
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        #---------------------------VFL启用2---------------------------------------------------------#
        # target_labels = target_labels.unsqueeze(-1).expand(-1, -1, self.nc) # self.nc: class num
        # one_hot = torch.zeros(target_labels.size(), device=self.device)
        # one_hot.scatter_(-1, target_labels, 1)
        # target_labels = one_hot
        #-----------------------------END--------------------------------------------------------------#

        # Cls loss   1.分类损失！！！！！！！！！！！！！！！！！！！
        #--------------------------VFL启用3--------------------------------------#
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #---------------------------END------------------------------------------#

        #-------------------------------------------------------------------------#
        #  pred_scores     ：torch.Size([16, 8400, 80])  预测框的类别预测信息  
        #  target_scores   ：torch.Size([16, 8400, 80])  将范围限制在正样本上 通过 one-hot编码后又用 软标签 来标记预测框的类
        #-------------------------------------------------------------------------#
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        #  正负样本都参加分类损失的计算 (不进行预测的框也要计算，不管在不在gt里面)




        # Bbox loss   2.回归损失！！！！！！！！！！！！！！！！！！
        #               (CIOU + DFL)
        #----------------------------------------------------------------------------------------------------#
        #  pred_distri            ：torch.Size([16, 8400, 64]) 
        #                           预测值位置值

        #  pred_bboxes            : torch.Size([16, 8400, 4])
        #                           由pred_distri进行离散变化变为xyxy(不是原图大小)

        #  anchor_points          : torch.Size([8400, 2])
        #                           生成每个特征图的中心点坐标(x,y)(不是原图大小)

        #  target_bboxes          : torch.Size([16, 8400, 4])
        #                           每个预测框对应 gt 的xyxy

        #  target_scores          : torch.Size([16, 8400, 80])
        #                           将范围限制在正样本上 通过 one-hot编码后又用软标签 来标记预测框的类

        #  target_scores_sum      : target_scores.sum()
        #                           

        #  fg_mask                : torch.Size([16, 8400])  
        #                           fg_mask代表的是有哪些锚点也就是预测框为1  也就是可以得到哪些预测框匹配到了gt（即哪些预测框为正样本）
        #-------------------------------------------------------------------------------------------------------#
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)





#---------------------------------------------------change----------------------------------------------------------------------#
    # def __call__(self, p, batch):
    #     loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    #     feats = p[1][0] if isinstance(p, tuple) else p[0]
    #     feats2 = p[1][1] if isinstance(p, tuple) else p[1]
        
    #     pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
    #         (self.reg_max * 4, self.nc), 1)
    #     pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    #     pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
    #     pred_distri2, pred_scores2 = torch.cat([xi.view(feats2[0].shape[0], self.no, -1) for xi in feats2], 2).split(
    #         (self.reg_max * 4, self.nc), 1)
    #     pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
    #     pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()

    #     dtype = pred_scores.dtype
    #     batch_size, grid_size = pred_scores.shape[:2]
    #     imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    #     anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    #     # targets
    #     targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  
    #     targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    #     gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    #     mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

    #     # pboxes
    #     pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
    #     pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, (b, h*w, 4)

    #     target_labels, target_bboxes, target_scores, fg_mask,_ = self.assigner(
    #         pred_scores.detach().sigmoid(),
    #         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    #         anchor_points * stride_tensor,
    #         gt_labels,
    #         gt_bboxes,
    #         mask_gt)
    #     target_labels2, target_bboxes2, target_scores2, fg_mask2,_ = self.assigner2(
    #         pred_scores2.detach().sigmoid(),
    #         (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
    #         anchor_points * stride_tensor,
    #         gt_labels,
    #         gt_bboxes,
    #         mask_gt)

    #     target_bboxes /= stride_tensor
    #     target_scores_sum = max(target_scores.sum(), 1)
    #     target_bboxes2 /= stride_tensor
    #     target_scores_sum2 = max(target_scores2.sum(), 1)

    #     # cls loss
    #     # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    #     loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # BCE
    #     loss[1] *= 0.25
    #     loss[1] += self.bce(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2 # BCE

    #     # bbox loss
    #     if fg_mask.sum():
    #         loss[0], loss[2]= self.bbox_loss(pred_distri,
    #                                                pred_bboxes,
    #                                                anchor_points,
    #                                                target_bboxes,
    #                                                target_scores,
    #                                                target_scores_sum,
    #                                                fg_mask)
    #         loss[0] *= 0.25
    #         loss[2] *= 0.25
    #     if fg_mask2.sum():
    #         loss0_, loss2_= self.bbox_loss2(pred_distri2,
    #                                                pred_bboxes2,
    #                                                anchor_points,
    #                                                target_bboxes2,
    #                                                target_scores2,
    #                                                target_scores_sum2,
    #                                                fg_mask2)
    #         loss[0] += loss0_
    #         loss[2] += loss2_

    #     loss[0] *= 7.5  # box gain
    #     loss[1] *= 0.5  # cls gain
    #     loss[2] *= 1.5  # dfl gain

    #     return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
#---------------------------------------------------end----------------------------------------------------------------------#

#-------------------------------------------------end------------------------------------------------------------------------------#



class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=coco8-obb.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
