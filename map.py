import torch
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from model import YOLOv2
from dataset import YOLOv2Dataset, collate_fn
from post_processing import post_processing
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