import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import YOLOv2
from loss import YOLOv2Loss
from dataset import YOLOv2Dataset, collate_fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchmetrics
detect_angle=True
num_classes=10
batch_size=1
num_workers=0
pre_trained_model_path = "./weights/ryolov2_model_299.pth"
# Define transformation for the validation dataset
transform_val = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
val_dataset = YOLOv2Dataset(root='./data', split='test', transform=transform_val, detect_angle=detect_angle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
model = YOLOv2(num_classes, detect_angle=detect_angle)

model.load_state_dict(torch.load(pre_trained_model_path, map_location=lambda storage, loc: storage))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
def post_processing(pred, anchors, conf_threshold=0.5, nms_threshold=0.1, detect_angle=detect_angle):
    
    num_anchors = len(anchors)
    anchors = torch.tensor(anchors)

    batch = pred.size(0)
    h = pred.size(2)
    w = pred.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w).to(device)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w).to(device)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1).to(device)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1).to(device)
    if detect_angle:
        anchor_a = anchors[:, 2].contiguous().view(1, num_anchors, 1).to(device)
    pred = pred.view(batch, num_anchors, -1, h * w)
    pred[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)
    pred[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)
    pred[:, :, 2, :].exp_().mul_(anchor_w).div_(w)
    pred[:, :, 3, :].exp_().mul_(anchor_h).div_(h)
    pred[:, :, 4, :].sigmoid_()
    if detect_angle:
        pred[:, :, 5, :].add_(anchor_a)

    cls_scores = torch.nn.functional.softmax(pred[:, :, 6:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    cls_max.mul_(pred[:, :, 4, :])

    score_thresh = cls_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.tensor([]))
    else:
        coords = pred.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        angle=torch.zeros_like(scores)
        if detect_angle:
            angle = pred.transpose(2, 3)[..., 5][score_thresh]
        detections = torch.cat([coords, scores[:, None], angle[:, None], idx[:, None]], dim=1)
        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.tensor([score_thresh_flat[s].int().sum() for s in slices], dtype=torch.int32)
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]
        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)
        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (does not consider class)
        conflicting = (ious > nms_threshold).triu(1)
        keep = conflicting.sum(0).byte()
        keep = keep.to(device)
        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 7).contiguous())

    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            final_boxes.append([[box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item(), box[5].item(), 
                                 int(box[6].item())] for box in boxes])
    return final_boxes
pred_list=[]
target_list=[]
with torch.no_grad():
    import math
    for val_images, val_targets in val_loader:
        # Forward pass
        val_outputs = model(val_images)
        predictions = post_processing(val_outputs, model.anchors, detect_angle=detect_angle)
        for b, (prediction, val_image, val_target) in enumerate(zip(predictions, val_images, val_targets)):

            # output_image = val_image.cpu().permute(1, 2, 0)
            # fig, ax = plt.subplots(1)
            # ax.imshow(output_image)
            bboxes=[]
            scores=[]
            labels=[]
            for pred in prediction:
                x, y, w, h, conf, angle, class_label = pred
                
                bboxes.append([x,y,w,h])
                scores.append(conf)
                labels.append(class_label)
                x = x * val_image.shape[1]
                y = y * val_image.shape[2]
                w = w * val_image.shape[1]
                h = h * val_image.shape[2]
                # Create a rotated rectangle patch
            #     rect = patches.Rectangle(
            #         (x - w / 2, y - h / 2), w, h, angle= -angle * 180 / math.pi,
            #         linewidth=2, edgecolor='r', facecolor='none', rotation_point = 'center'
            #     )

            #     arrow = patches.Arrow(x, y, w / 2 * math.cos(-angle - math.pi/2), w / 2 * math.sin(-angle - math.pi/2), color="green", linewidth=2)

            #     # Add the patch to the Axes
            #     ax.add_patch(rect)
            #     ax.add_patch(arrow)

            #     # Annotate with class label
            #     ax.text(x - w / 2, y - h / 2, f'{class_label}, {conf:.2f}', color='r', fontsize=12, va='bottom', ha='left')
            # plt.show()
            pred_list.append({"boxes":torch.tensor(bboxes, device=device),
                                    "scores":torch.tensor(scores, device=device),
                                    "labels":torch.tensor(labels, dtype=torch.int32, device=device)})
            val_target = torch.stack(val_target, dim=0)
            target_list.append({"boxes":val_target[...,:4],
                                    "labels":val_target[...,5].to(torch.int32)})
metric = torchmetrics.detection.mean_ap.MeanAveragePrecision(box_format='cxcywh')
metric.update(pred_list, target_list)
res = metric.compute()
print(res['map_50'].item()*100, res['map_75'].item()*100)