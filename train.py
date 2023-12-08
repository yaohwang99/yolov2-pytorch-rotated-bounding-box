import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import YOLOv2
from loss import YOLOv2Loss
from dataset import YOLOv2Dataset, collate_fn
num_classes=10
batch_size=4
num_workers=0
num_epochs = 500
trained_epoch = 300
# Load your trained YOLOv2 model
model = YOLOv2(num_classes=num_classes)  # Adjust the number of classes and boxes accordingly
model.load_state_dict(torch.load('yolov2_model_300.pth'))  # Load the trained weights

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
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, targets in train_loader:
            # Forward pass
            outputs = model(images)
            # Compute loss
            loss = criterion(outputs, targets)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.update(1)

    avg_train_loss = running_loss / len(train_loader)
    print(f'Train Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            # Forward pass
            val_outputs = model(val_images)

            # Compute validation loss
            val_loss += criterion(val_outputs, val_targets).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Log losses to TensorBoard
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), 'yolov2_model_' + str(epoch + 1 + trained_epoch) + '.pth')
# Save the trained model
torch.save(model.state_dict(), 'yolov2_model.pth')

# Close the TensorBoard writer
writer.close()