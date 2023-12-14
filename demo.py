import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import YOLOv2
from loss import YOLOv2Loss
from dataset import YOLOv2Dataset, collate_fn
from post_processing import post_processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
num_classes=10
batch_size=4
num_workers=0
pre_trained_model_path = "./weights/ryolov2_model_316.pth"
# Define transformation for the validation dataset
transform_val = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
val_dataset = YOLOv2Dataset(root='./data', split='test', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
model = YOLOv2(num_classes)

model.load_state_dict(torch.load(pre_trained_model_path, map_location=lambda storage, loc: storage))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
with torch.no_grad():
    import math
    i = 0
    for val_images, val_targets in val_loader:
        # Forward pass
        val_outputs = model(val_images)
        predictions = post_processing(val_outputs, model.anchors)
        for b, (prediction, val_image) in enumerate(zip(predictions, val_images)):

            output_image = val_image.cpu().permute(1, 2, 0)
            fig, ax = plt.subplots(1)
            ax.imshow(output_image)
            for pred in prediction:
                x, y, w, h, conf, angle, class_label = pred
                x = x * val_image.shape[1]
                y = y * val_image.shape[2]
                w = w * val_image.shape[1]
                h = h * val_image.shape[2]

                # Create a rotated rectangle patch
                rect = patches.Rectangle(
                    (x - w / 2, y - h / 2), w, h, angle= -angle * 180 / math.pi,
                    linewidth=2, edgecolor='r', facecolor='none', rotation_point = 'center'
                )

                arrow = patches.Arrow(x, y, w / 2 * math.cos(-angle - math.pi/2), w / 2 * math.sin(-angle - math.pi/2), color="green", linewidth=2)

                # Add the patch to the Axes
                ax.add_patch(rect)
                ax.add_patch(arrow)

                # Annotate with class label
                ax.text(x - w / 2, y - h / 2, f'{class_label}, {conf:.2f}', color='r', fontsize=12, va='bottom', ha='left')
            # plt.savefig(f"./tmp/file%02d.png" % i)
            i+=1
            plt.show()
# import os
# import subprocess
# import glob
# os.chdir("./tmp")
# subprocess.call([
#     'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
#     'video_name.mp4'
# ])
# for file_name in glob.glob("*.png"):
#     os.remove(file_name)