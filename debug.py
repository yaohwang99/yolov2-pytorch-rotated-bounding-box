import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import YOLOv2
from loss import YOLOv2Loss
from dataset import YOLOv2Dataset, collate_fn
num_classes=10
batch_size=4
num_workers=0
num_epochs = 500
# Load your trained YOLOv2 model
model = YOLOv2(num_classes=num_classes, detect_angle=True)  # Adjust the number of classes and boxes accordingly
# model.load_state_dict(torch.load('yolov2_model.pth'))  # Load the trained weights

# Define transformation for the training dataset
transform_train = transforms.Compose([transforms.Resize((416, 416)),
                                      transforms.ToTensor()])

# Define transformation for the validation dataset
transform_val = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])

# Load VOC training dataset
train_dataset = YOLOv2Dataset(root='./data', split='obj', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

# Load VOC validation dataset
val_dataset = YOLOv2Dataset(root='./data', split='test', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


# Initialize YOLOv2 model, loss function, and optimizer
criterion = YOLOv2Loss(model.num_classes, model.anchors)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set up TensorBoard writer
writer = SummaryWriter()

# Training loop
assert torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Training phase
model.train()
running_loss = 0.0
for images, targets in train_loader:
    # Forward pass
    outputs = model(images)
    # Compute loss
    loss = criterion(outputs, targets)
    print("done")
    break