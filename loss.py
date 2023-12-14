import torch
import torch.nn as nn
import math
class YOLOv2Loss(nn.Module):
    def __init__(self, num_classes, anchors, coord_scale=5.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, theta_scale=5.0):
        super(YOLOv2Loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = torch.tensor(anchors)

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.theta_scale = theta_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, output, target):

        batch_size = output.shape[0]
        height = output.shape[2]
        width = output.shape[3]

        # Get x,y,w,h,a,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width) 
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()  
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]
        conf = output[:, :, 4, :].sigmoid()
        theta = output[:, :, 5, :]
        cls = output[:, :, 6:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                    height * width).transpose(1, 2).contiguous().view(-1,
                                                                                                      self.num_classes)
        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, ttheta = self.build_targets(target, height, width)

        theta = theta[cls_mask].view(-1)
        ttheta = ttheta[cls_mask].view(-1)
        coord_mask = coord_mask.expand_as(tcoord)
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)
        cls = cls[cls_mask].view(-1, self.num_classes)
        
        # Compute losses
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        sl = nn.SmoothL1Loss()
        self.loss_coord = self.coord_scale * sl(coord * coord_mask, tcoord * coord_mask)
        self.loss_conf = sl(conf * conf_mask, tconf * conf_mask)
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls)
        self.loss_theta = self.theta_scale * sl(theta, ttheta)
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls + self.loss_theta

        return self.loss_tot, {"coord": self.loss_coord.item(), "conf": self.loss_conf.item(), 
                               "cls": self.loss_cls.item(), "theta": self.loss_theta.item()}

    def build_targets(self, ground_truth, height, width):
        batch_size = len(ground_truth)
        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False, device=self.device) * self.noobject_scale
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False, device=self.device).bool()
        cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False, device=self.device).bool()
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False, device=self.device)
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False, device=self.device)
        tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False, device=self.device)
        ttheta = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False, device=self.device)
        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            anchors = torch.cat([torch.zeros_like(self.anchors[...,:2]), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 5)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = anno[0] * width
                gt[i, 1] = anno[1] * height
                gt[i, 2] = anno[2] * width
                gt[i, 3] = anno[3] * height
                gt[i, 4] = anno[4] * math.pi / 8 # angle in radian

            # Find best anchor for each ground truth
            gt_expanded = gt[..., 4].repeat(self.num_anchors, 1)
            # Expand anchors to have the same shape as gt_expanded
            anchors_expanded = torch.unsqueeze(anchors[...,4], 1).repeat(1, len(ground_truth[b]))

            # Calculate best anchor
            iou_gt_anchors = torch.cos(gt_expanded * 0.25 - anchors_expanded * 0.25)
            _, best_anchors = iou_gt_anchors.max(0)
            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_anchors[best_n][i]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                ttheta[b][best_n][gj * width + gi] = gt[i, 4] - anchors[best_n][4]
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[5])
                
        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, ttheta