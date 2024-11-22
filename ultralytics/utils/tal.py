# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")

#--------------------------------------------------------------------------------------------#
#  作用               ：匹配正样本 通过 align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
#                       即将grid_cell中心点在gt框内部生成的预测框 与 gt框进行类别分数计算（bbox_scores.pow(self.alpha)）和 iou计算（overlaps.pow(self.beta)）
#                       选则 align_metric 分数在前topk（10） 个预测框 作为正样本 ，对于一个预测框匹配多个gt框，选择与gt框更少的iou进行匹配
#
#  返回     
#  target_labels      ：torch.Size([16, 8400])
#                       每个预测框的对应 gt 的类别

#  target_bboxes      ：torch.Size([16, 8400, 4])
#                       每个预测框对应 gt 的xyxy

#  target_scores      ：torch.Size([16, 8400, 80])
#                       将范围限制在正样本上 通过 one-hot编码后又用软标签 来标记预测框的类

#  fg_mask.bool()     ：torch.Size([16, 8400])
#                       fg_mask代表的是有哪些锚点也就是预测框为1  也就是可以得到哪些预测框匹配到了gt（即哪些预测框为正样本）

#  target_gt_idx      ：torch.Size([16, 8400])
#                       找到每一个预测框对应的gt框的索引

#--------------------------------------------------------------------------------------------#
class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    #-------------------------------------------------------------------------------------------------------------------#
    #  pred_scores.detach().sigmoid()              ：torch.Size([16, 8400, 80])     预测框的类别预测信息
    #  (pred_bboxes.detach() * stride_tensor)      ：torch.Size([16, 8400, 4])      将预测的xyxy还原为真实图像的xyxy
    #  anchor_points * stride_tensor               ：torch.Size([8400, 2])          将预测框的中心点还原为真实图像
    #  gt_labels                                   ：torch.Size([16, max_label, 1]) 标签框的类别信息
    #  gt_bboxes                                   : torch.Size([16, max_label, 4]) 标签框的xyxy信息
    #  mask_gt                                     : torch.Size([16, max_label])    选择有效标签框
    #-------------------------------------------------------------------------------------------------------------------#
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )
        #--------------------------------------------------------------------------------#
        #  mask_pos          ：torch.Size([16,37,8400])
        #                      选择有效真实框, 锚点落在真实框内部, 该锚点对应的预测框与真实框的匹配度是topk
        #                      就像一个表格  纵坐标为每张图片的label  横坐标为84000个预测框

        #  align_metric      ：torch.Size([16, 37, 8400]) 
        #                      由预测框与gt的分类得分的alpha次方 * 预测框与gt的iou构成

        #  overlaps          ：torch.Size([16, 37, 8400])
        #                      预测框与gt的iou构成 (当一个预测框 与 多个gt匹配时会将 预测框分配给iou更大的gt   此时会用到overlaps)
        #
        #                      gt框为真实有效的标签框，不是由0填充的
        #--------------------------------------------------------------------------------#
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        
        #-------------------------------------------------------------------------------#
        #  self.select_highest_overlaps  :处理一个预测框对应多个gt框，保留ciou值最大的真实框

        #  target_gt_idx                 ：torch.Size([16, 8400])
        #                                  找到每一个预测框对应的gt框的索引

        #  fg_mask                       : mask_pos.sum(-2)   torch.Size([16, 8400])
        #                                  fg_mask代表的是有哪些锚点也就是预测框为1  也就是可以得到哪些预测框匹配到了gt

        #  mask_pos                      : torch.Size([16,37,8400])
        #                                  就像一个表格  纵坐标为每张图片的label  横坐标为84000个预测框
        #-------------------------------------------------------------------------------#
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        #-----------------------------------------------------------------------------------------------------------#
        #  Assigned target

        #  target_labels          : torch.Size([16, 8400])     
        #                           每个预测框的对应 gt 的类别

        #  target_bboxes          : torch.Size([16, 8400, 4])
        #                           每个预测框对应 gt 的xyxy

        #  target_scores          : torch.Size([16, 8400, 80])
        #                           将范围限制在正样本上 通过 one-hot编码后又用软标签 来标记预测框的类
        #-----------------------------------------------------------------------------------------------------------#
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        
        #-------------------------------------------------#
        #  Normalize
        #  归一化标签(基于对齐分数t和CIoU的软标签)
        
        #  align_metric *= mask_pos  此时算的是grid_cell中点在真实gt内部，由这些grid_cell产生的预测框与gt 产生的align_metric值
        #--------------------------------------------------#
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj

        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)

        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        #----------------------------------------------------------------------------------#
        #  mask_in_gts  torch.Size([16, 37, 8400])   
        #  判断8400个anchor中心点与16x24个gt框的位置关系，anchor中心点是否在gt框里面，在的话为1，不在即为0
        #----------------------------------------------------------------------------------#
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) 
        
        #---------------------------------------------------------------------#
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # align_metric        ：torch.Size([16, 37, 8400])  
        #                       bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        #                       由预测框与gt的分类分值alpha次方 * 预测框与gt的iou构成

        # overlaps            ：torch.Size([16, 37, 8400])
        #                       预测框与gt的iou构成   
        
        #                      gt为真实的标签框，不是凑数填充为0的
        #---------------------------------------------------------------------#
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)

        #---------------------------------------------------------------------#
        # Get topk_metric mask, (b, max_num_obj, h*w)
        # mask_topk           ：torch.Size(16,37,8400)
        #                     :如mask_topk[16][21][8300]=1表示第32张图片第22个gt对应的是第8300个预测框, 这个预测框是正样本
        #                     :每个gt框 已将选择了<=self.topk 个预测框 作为正样本
        #---------------------------------------------------------------------#
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())

        #---------------------------------------------------------------------#
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        # mask_pos            ：torch.Size(16,37,8400)   就像一个表格  纵坐标为每张图片的label  横坐标为84000个预测框
        #                     ：选择有效真实框, 锚点落在真实框内部, 该锚点对应的预测框与真实框的匹配度是topk
        #---------------------------------------------------------------------
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        #--------------------------------------------------------------------------#
        #  pd_scores       ：torch.Size([16, 8400, 80])
        #  pd_bboxes       : torch.Size([16, 8400, 4])
        #  gt_labels       : torch.Size([16, max_label, 1])
        #  gt_bboxes       : torch.Size([16, max_label, 4])
        #  mask_gt         : torch.Size([16, max_label, 8400])
        #--------------------------------------------------------------------------#
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        #---------------------------------------------------------------#
        #  bbox_scores    ：torch.Size([16, 73, 8400])
        #                   获取预测框与gt框的分类值  
        #                   eg: 预测框有80个类 直接取出与gt框相同类的 得分

        #  overlaps       ：torch.Size([16, 73, 8400])
        #                   由预测框与gt的iou构成 
        #---------------------------------------------------------------#
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Iou calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        #-----------------------------------------------------------------#
        #  metrics：torch.Size([16, 28, 8400])    topk_mask：torch.Size([16, 28, 10])
        #-----------------------------------------------------------------#
        """   
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        #--------------------------------------------------------------------------------#
        # (b, max_num_obj, topk)    
        # topk_metrics    ：torch.Size([16, 28, 10])  存放（align_metric）前十的数值  
        # topk_idxs       ：torch.Size([16, 28, 10])  存放（align_metric）前十的数值所在序号
        #--------------------------------------------------------------------------------#
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        #-------------------------------------------------------------------------#
        # (b, max_num_obj, topk)    
        # topk_idxs.masked_fill_(~topk_mask, 0)    如果真实框无效 则置为0,因为 每张图片里面的的标签数都被强行写为max_label个,没有满max_label填充为0
        # topk_idxs       ：torch.Size([16, 28, 10])  
        #                 如topk_idxs[31][21][0]的值是8300,也就是说第32张图片中的第22个gt与第8300的pd匹配度是位于前10中
            
        # count_tensor    : torch.Size([16, 28, 8400]) 
        #                 count_tensor[31][21][8300]=1表示第32张图片第22个gt对应的是第8300个pd, 这个pd是正样本
        #-------------------------------------------------------------------------#
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)    (16,8400,4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores     # 将标签值限制在0-正无穷  小于的补位0
        target_labels.clamp_(0)      # (16,8400)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod     # xy_centers：shape(8400,2)   gt_bboxes：shape(16,24,4)
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom            
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1) #(16,24,8400,4)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps) #(16,24,8400)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoI will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        #--------------------------------------------------------------------------#
        #   mask_pos                fg_mask
        #  (b, n_max_boxes, h*w) -> (b, h*w)
        #  就像一个表格  纵坐标为每张图片的label  横坐标为84000个预测框

        #  当 fg_mask.max() > 1 则说明   预测框同时被分给多个gt 
        #--------------------------------------------------------------------------#
        fg_mask = mask_pos.sum(-2)

        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        #-----------------------------------------------------#
        #  target_gt_idx    :torch.Size([16，8400])  
        #                    找到每一个预测框 对应的 gt框的索引
        #------------------------------------------------------#
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Iou calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)
        return is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
