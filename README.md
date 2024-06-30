# Traffic Sign Recognition with YOLOv8
![sample output](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20194358.png)
-   [About](#about)
    -   [Recognized Traffic Signs](#recognized-traffic-signs)
    -   [Potential Applications](#potential-applications)
    -   [Dataset](#dataset)
-   [Step by Step](#step-by-step)
    -   [Installation](#installation)
    -   [Displaying Sample Images with Bounding Boxes](#displaying-sample-images-with-bounding-boxes)
    -   [Data Augmentation](#data-augmentation)
    -   [Training the Model](#training-the-model)
    -   [Evaluating the Model](#evaluating-the-model)
-   [Metric Charts](#metric-charts)


# About

This project demonstrates how to perform traffic sign recognition on a custom dataset using the YOLO (You Only Look Once) model. The process includes mounting Google Drive to access the dataset, displaying labeled images, augmenting the dataset, training the YOLO model, and validating the model's performance on test images.

## Recognized Traffic Signs
The system recognizes the following traffic signs:
-   Direction Mandatory
-   Speed Limit
-   Weight Limit
-   Parking
-   Pedestrian Crossing
-   Bus Stop
-   Roundabout
-   Stop
-   Yield
-   Warning
-   No Parking
-   No Entry
-   No Turn
-   No Stopping
-   No Vehicles

## Potential Applications

The deployment of a traffic sign recognition system could be beneficial in several areas:
-   **Driver Assistance**: A navigation aid system could continuously inform the driver about the passed signs and alert them with an audible signal if, for example, a speed limit is not followed. It could also facilitate navigation in unfamiliar cities, reducing driver stress and fatigue.
    
-   **Road Condition Alerts**: Vehicles passing through marked roadwork areas could send notifications to nearby devices about traffic disruptions.
    
-   **Support for Autonomous Vehicles**: The system would enable vehicles to make appropriate decisions on the road without human intervention.
    
-   **Data for Road Administration**: Aggregated data could assist in planning for safety improvements, helping officials identify problematic areas for drivers.
    
-   **Monitoring Professional Drivers**: Company owners transporting, for example, school trips, could check if the driver adheres to traffic signs.
    
-   **Personal Driving Statistics**: Individuals could review their driving statistics and work on improving safe driving habits.
    
-   **Traffic Sign Learning Support**: Helpful for driver’s license courses by supporting traffic sign learning.

## Dataset

Screenshots from Google Maps, taken on roads in Lublin, each with dimensions of 1200 by 900 px, were labeled using the Label Studio program. The bounding box coordinates are in Pascal VOC format, meaning they are recorded as (xmin, ymin, xmax, ymax).

The images and labels were then downloaded in a format accepted by the YOLOv8 model (labels and images in separate folders) and divided into training and test sets in a ratio of 8 to 2.

# Step by step

This section provides a detailed guide and explanation of the code used in this project.

## Installation

Ensure your dataset is stored in your Google Drive and mount it.
```python
from google.colab import drive
drive.mount('/content/drive')
```
Install Required Libraries:
```
!pip install opencv-python-headless matplotlib albumentations ultralytics
```
## Displaying Sample Images with Bounding Boxes

```python
import cv2
import os
import matplotlib.pyplot as plt
from IPython.display import display

def draw_bounding_boxes(image, label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        img_height, img_width = image.shape[:2]
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

for image_name in os.listdir(images_path)[1:10]:
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(images_path, image_name)
        label_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Label does not exist: {label_path}")
            continue

        image_with_boxes = draw_bounding_boxes(image, label_path)

        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        display(plt.gcf())

```
![sample](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20194132.png)
![sample](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20194153.png)

## Data Augmentation

```python
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

image_dir = '/content/drive/My Drive/projekt_znaki/train/images'
label_dir = '/content/drive/My Drive/projekt_znaki/train/labels'
aug_image_dir = '/content/drive/My Drive/projekt_znaki/train/augmented/images'
aug_label_dir = '/content/drive/My Drive/projekt_znaki/train/augmented/labels'

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

transform = A.Compose([
    A.Rotate(limit=45, p=1.0),
    A.RandomBrightnessContrast(p=0.2)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

for image_name in os.listdir(image_dir):
    if image_name.endswith('.png'):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        label_path = os.path.join(label_dir, image_name.replace('.png', '.txt'))
        with open(label_path, 'r') as file:
            bboxes = []
            class_labels = []
            for line in file.readlines():
                elements = line.split()
                class_labels.append(int(elements[0]))
                bbox_values = list(map(float, elements[1:]))
                x_center, y_center, width, height = bbox_values
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height
                bboxes.append([x_min, y_min, x_max, y_max])

        for i in range(5):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            aug_image_name = image_name.replace('.png', f'_aug_{i}.png')
            aug_image_path = os.path.join(aug_image_dir, aug_image_name)

            cv2.imwrite(aug_image_path, aug_image)

            aug_label_path = os.path.join(aug_label_dir, aug_image_name.replace('.png', '.txt'))
            with open(aug_label_path, 'w') as file:
                for bbox, label in zip(aug_bboxes, aug_labels):
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                    file.write(f"{label} {' '.join(map(str, yolo_bbox))}\n")
```
![LAugmentation](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20200347.png)

## Training the Model
Using the _yolov8_config.yaml_ file traing the model with the code below.

```python
import os
import yaml
from ultralytics import YOLO

with open('/content/drive/My Drive/projekt_znaki/yolov8_config.yaml') as f:
    config = yaml.safe_load(f)

model = YOLO(config['model']['type'])

data = '/content/drive/My Drive/projekt_znaki/yolov8_config.yaml'
model.train(data=data, epochs=50, imgsz=[1200, 900])
```
_yolov8_config.yaml_:
```yaml
train: /content/drive/My Drive/projekt_znaki/train/augmented
val: /content/drive/My Drive/projekt_znaki/val
model:
	type: yolov8n
	anchors:
	- [39, 43, 72, 84, 146, 148]
	- [15, 16, 629, 786, 722, 723]
	- [783, 783, 910, 805, 823, 823]
	nc: 19
	names: ['droga_z_pierwszenstwem', 'dzieci', 'jednokierunkowa', 'koniec_pierwszenstwa', 'nakaz_jazdy', 'ograniczenie_predkosci', 'ograniczenie_tonazu', 'parking', 'przejscie_dla_pieszych', 'przystanek', 'rondo', 'stop', 'ustap_pierwszenstwa', 'uwaga', 'zakaz_postoju', 'zakaz_ruchu', 'zakaz_skretu', 'zakaz_wjazdu', 'zakaz_zatrzymywania']
```
![sample output](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20194437.png)
![sample output](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/screenshots/Zrzut%20ekranu%202024-06-30%20194448.png)
## Evaluating the Model

Model Evaluation on Test Images:
```python
from PIL import Image
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import os

model = YOLO('/content/drive/My Drive/projekt_znaki/runs/detect/train2/weights/best.pt')

images_path = '/content/drive/My Drive/projekt_znaki/obrazkiDoTestow'

image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
print(image_files)

for image_file in image_files:
    image_path = os.path.join(images_path, image_file)

    image = cv2.imread(image_path)
    results = model(image)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{name}: {confidence:.2f}"
        cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2_imshow(image)
```
# Metric Charts

The following charts depict various training and validation metrics for the model.
![charts](https://raw.githubusercontent.com/Mar0u/ai-recognition-road-signs/main/runs/detect/train2/results.png)
Here's an analysis of each metric shown:

-   **train/box_loss**: This represents the loss related to bounding box fitting during training. The decreasing trend indicates that the model is improving in predicting the bounding box positions in the training data.
    
-   **train/cls_loss**: This is the classification loss during training. The downward trend suggests that the model is getting better at classifying the detected objects.
    
-   **train/dfl_loss**: This refers to the distribution focal loss, indicating the accuracy of boundary predictions. Similar to the other losses, its decline implies enhanced model performance.
    
-   **metrics/precision(B)**: Precision during training. The increasing value of precision suggests that the model is getting better at avoiding false positives.
    
-   **metrics/recall(B)**: Recall during training. The upward trend indicates that the model is improving in detecting true positive instances.
    
-   **metrics/mAP50(B)**: Mean Average Precision (mAP) at 50% Intersection over Union (IoU). The rise in this metric indicates that the model is performing well in object detection.
    
-   **metrics/mAP50-95(B)**: Average mAP for various IoU values. Although this metric is also increasing, it does so more slowly, suggesting that the model may struggle with higher IoU thresholds, indicating difficulties in precisely detecting object boundaries or finding occluded objects.
    
-   **val/box_loss**: Validation loss for bounding box fitting. The decreasing trend, despite fluctuations, indicates improving performance on the validation set.
    
-   **val/cls_loss**: Validation classification loss. The decrease signifies improved classification performance on unseen data.
    
-   **val/dfl_loss**: Validation distribution focal loss. A decreasing trend, though fluctuating, suggests better boundary prediction on the validation set.
    

Overall, losses in both the training and validation sets are decreasing, indicating that the model is learning and improving. However, the validation losses are higher than the training losses, which could suggest slight overfitting. To mitigate this issue, adding more images to the dataset might be beneficial.
