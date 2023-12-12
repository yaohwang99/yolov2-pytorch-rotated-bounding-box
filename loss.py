import torch
import torch.nn as nn
import math
class YOLOv2Loss(nn.Module):
    def __init__(self, num_classes, anchors, coord_scale=5.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, theta_scale=5.0, thresh=0.6):
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
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()  
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]
        conf = output[:, :, 4, :].sigmoid()
        theta = output[:, :, 5, :]
        cls = output[:, :, 6:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                    height * width).transpose(1, 2).contiguous().view(-1,
                                                                                                      self.num_classes)
        # print(coord.shape, conf.shape, cls.shape)
        # torch.Size([4, 5, 4, 169]) torch.Size([4, 5, 169]) torch.Size([3380, 10])
        # Create prediction boxes
        # pred_boxes = torch.zeros([batch_size * self.num_anchors * height * width, 5])
        lin_x = torch.arange(0, width).repeat(height, 1).view(height * width)
        lin_y = torch.arange(0, height).repeat(width, 1).t().contiguous().view(height * width)
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)
        anchor_a = self.anchors[:, 2].contiguous().view(self.num_anchors, 1)
        # print("check 2:", pred_boxes.shape, lin_x, lin_y, anchor_w, anchor_h)
        # torch.Size([3380, 4])
        if torch.cuda.is_available():
            # pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()
            anchor_a = anchor_a.cuda()

        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, ttheta = self.build_targets(None, target, height, width)
        
        # print("check 4:", coord_mask.shape, conf_mask.shape, cls_mask.shape, tcoord.shape, tconf.shape, tcls.shape)
        # torch.Size([4, 5, 1, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 4, 169]) torch.Size([4, 5, 169]) torch.Size([4, 5, 169])
        theta = theta[cls_mask].view(-1)
        ttheta = ttheta[cls_mask].view(-1)
        coord_mask = coord_mask.expand_as(tcoord)
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)
        # print("check 5:", coord_mask.shape, cls_mask.shape, tcls.shape)
        # torch.Size([4, 5, 4, 169]) torch.Size([3380, 10]) torch.Size([4])
        cls = cls[cls_mask].view(-1, self.num_classes)
        
        # Compute losses
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        sl = nn.SmoothL1Loss()
        # self.loss_coord = self.coord_scale * kfiou_loss(coord[coord_mask].view(-1,5), tcoord[coord_mask].view(-1, 5)).sum(0)
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask)
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask)
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls)
        self.loss_theta = self.theta_scale * sl(theta, ttheta)
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls + self.loss_theta

        return self.loss_tot, {"coord": self.loss_coord.item(), "conf": self.loss_conf.item(), 
                               "cls": self.loss_cls.item(), "theta": self.loss_theta.item()}

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)
        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False).bool()
        cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).bool()
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        ttheta = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            tcls = tcls.cuda()
            cls_mask = cls_mask.cuda()
            ttheta = ttheta.cuda()
        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            # cur_pred_boxes = pred_boxes[
            #                  b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
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
            # Expand anchors to have the same shape as gt_wh_clone
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