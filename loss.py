import torch
import torch.nn as nn
import math
class YOLOv2Loss(nn.Module):
    def __init__(self, num_classes, anchors, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, theta_scale=1.0, thresh=0.6):
        super(YOLOv2Loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = torch.tensor(anchors)

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.theta_scale = theta_scale
        self.thresh = thresh

    def forward(self, output, target):

        batch_size = output.shape[0]
        height = output.shape[2]
        width = output.shape[3]

        # Get x,y,w,h,a,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width) 
        coord = torch.zeros_like(output[:, :, :5, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()  
        coord[:, :, 2:5, :] = output[:, :, 2:5, :]
        conf = output[:, :, 4, :].sigmoid()
        theta = output[:, :, 5, ]
        cls = output[:, :, 6:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                    height * width).transpose(1, 2).contiguous().view(-1,
                                                                                                      self.num_classes)
        # print(coord.shape, conf.shape, cls.shape)
        # torch.Size([4, 5, 4, 169]) torch.Size([4, 5, 169]) torch.Size([3380, 10])
        # Create prediction boxes
        pred_boxes = torch.zeros([batch_size * self.num_anchors * height * width, 5])
        lin_x = torch.arange(0, width).repeat(height, 1).view(height * width)
        lin_y = torch.arange(0, height).repeat(width, 1).t().contiguous().view(height * width)
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)
        anchor_a = self.anchors[:, 2].contiguous().view(self.num_anchors, 1)
        # print("check 2:", pred_boxes.shape, lin_x, lin_y, anchor_w, anchor_h)
        # torch.Size([3380, 4])
        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()
            anchor_a = anchor_a.cuda()


        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes[:, 4] = (coord[:, :, 4].detach() + anchor_a).view(-1)
        pred_boxes = pred_boxes.cpu()
        # print("check 3:", pred_boxes.shape)
        # torch.Size([3380, 4])
        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, ttheta = self.build_targets(pred_boxes, target, height, width)
        # print("check 4:", coord_mask.shape, conf_mask.shape, cls_mask.shape, tcoord.shape, tconf.shape, tcls.shape)
        # torch.Size([4, 5, 1, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 4, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 169])
        coord_mask = coord_mask.expand_as(tcoord)
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)
        # print("check 5:", coord_mask.shape, cls_mask.shape, tcls.shape)
        # torch.Size([4, 5, 4, 169]) torch.Size([3380, 10]) torch.Size([4])
        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            tcls = tcls.cuda()
            cls_mask = cls_mask.cuda()
            ttheta = ttheta.cuda()

        conf_mask = conf_mask.sqrt()
        cls = cls[cls_mask].view(-1, self.num_classes)
        # Compute losses
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        bce = nn.BCEWithLogitsLoss()
        self.loss_coord = self.coord_scale * kfiou(coord[coord_mask].view(-1,5), tcoord[coord_mask].view(-1, 5),
                                                   coord[coord_mask].view(-1,5), tcoord[coord_mask].view(-1, 5)).sum(0)
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask)
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls)
        self.loss_theta = self.theta_scale * bce(theta, ttheta)
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls

        return self.loss_tot

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)

        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False).bool()
        cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).bool()
        tcoord = torch.zeros(batch_size, self.num_anchors, 5, height * width, requires_grad=False)
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        ttheta = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[
                             b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
            anchors = torch.cat([torch.zeros_like(self.anchors[...,:2]), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 5)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = anno[0] * width
                gt[i, 1] = anno[1] * height
                gt[i, 2] = anno[2] * width
                gt[i, 3] = anno[3] * height
                gt[i, 4] = anno[4] * math.pi / 8 # angle in radius

            # Set confidence mask of matching detections to 0
            # iou_gt_pred = kfiou(gt, cur_pred_boxes, gt, cur_pred_boxes)
            # mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            # conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each ground truth
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            gt_wh_clone = torch.unsqueeze(gt_wh, 1).repeat(1, self.num_anchors, 1)

            # Expand anchors to have the same shape as gt_wh_clone
            anchors_expanded = torch.unsqueeze(anchors, 0).expand_as(gt_wh_clone)
            # Calculate IoU between gt_wh_clone and anchors_expanded using kfiou function
            iou_gt_anchors = kfiou(gt_wh_clone, anchors_expanded, gt_wh_clone, anchors_expanded)

            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_anchors[i][best_n]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tcoord[b][best_n][4][gj * width + gi] = gt[i, 4]
                ttheta[b][best_n][gj * width + gi] = gt[i, 4]
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[5])
                
        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, ttheta


def bbox_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections + 10e-15

    return intersections / unions

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def kfiou(pred,
        target,
        pred_decode=None,
        targets_decode=None,
        fun=None,
        beta=1.0 / 9.0,
        eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[..., :2]
    xy_t = target[..., :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU
    kf_loss = kf_loss.reshape_as(xy_loss)

    loss = (xy_loss + kf_loss).clamp(0)
    return loss