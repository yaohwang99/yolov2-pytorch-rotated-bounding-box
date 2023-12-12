import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
# Function to parse the label from the text file
def parse_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        label = list(map(float, line.strip().split()))
        labels.append(label)

    return labels

# Function to draw bounding boxes on the image
def draw_boxes(image_path, labels):
    # Read the image
    img = plt.imread(image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Draw bounding boxes
    for label in labels:
        x, y, w, h, angle, class_label = label
        x = x * img.shape[0]
        y = y * img.shape[1]
        w = w * img.shape[0]
        h = h * img.shape[1]
        # Create a rotated rectangle patch
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2), w, h, angle=angle * 360 / 16,
            linewidth=2, edgecolor='r', facecolor='none', rotation_point = 'center'
        )
        arrow = patches.Arrow(x, y, w / 2, h / 2, color="green", linewidth=2)

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(arrow)
        # Annotate with class label
        ax.text(x, y, f'{class_label}', color='r', fontsize=8, va='bottom', ha='left')

    # Show the plot
    plt.show()

# Example usage
image_path = 'data/test/color_93_2.jpg'
label_path = 'data/test/color_93_2.txt'

labels = parse_label(label_path)
draw_boxes(image_path, labels)