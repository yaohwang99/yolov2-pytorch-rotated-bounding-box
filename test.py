import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import YOLOv2

# Load your trained YOLOv2 model
model = YOLOv2(num_classes=20, num_boxes=5)  # Adjust the number of classes and boxes accordingly
model.load_state_dict(torch.load('yolov2_model.pth'))  # Load the trained weights
model.eval()  # Set the model to evaluation mode

# Define transformation for the test/validation dataset
transform = transforms.Compose([transforms.Resize((416, 416)),
                                transforms.ToTensor()])

# Load VOC test/validation dataset
test_dataset = datasets.VOCDetection(root='./data', year='2012', image_set='val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Test/Validation loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)

        # Assuming classification is the first part of the output
        pred_classes = torch.argmax(outputs[:, :20], dim=1)
        true_classes = targets[:, 0].long()

        total_correct += torch.sum(pred_classes == true_classes).item()
        total_samples += targets.size(0)

accuracy = total_correct / total_samples
print(f'Accuracy on test/validation set: {accuracy * 100:.2f}%')