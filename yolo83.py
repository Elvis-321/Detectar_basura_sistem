import cv2 as cv
import numpy as np
from ultralytics import YOLO
from screeninfo import get_monitors
import os
import datetime

# Initialize video capture
cap = cv.VideoCapture('Basura.mp4')
confThreshold = 0.5
nmsThreshold = 0.2

# Load YOLOv8 models
model1 = YOLO('./runs/detect/train5/weights/best.pt')  # TACO model
model2 = YOLO('yolov8n.pt')  # COCO model

# Load TACO names
with open("taco.names", 'rt') as f:
    classNames1 = f.read().strip().split('\n')

# Load COCO names
with open("coco.names", 'rt') as f:
    classNames2 = f.read().strip().split('\n')

print("TACO classes:", classNames1)
print("COCO classes:", classNames2)

# Create directory for report
if not os.path.exists('REPORTE'):
    os.makedirs('REPORTE')

# Get screen resolution
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Calculate the target window size (70% of screen size)
target_width = int(screen_width * 0.7)
target_height = int(screen_height * 0.7)

# Set window name
window_name = 'Detector de basura'

# Set the window size
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

def resize_with_aspect_ratio(image, width):
    """ Resize image maintaining the aspect ratio """
    aspect_ratio = image.shape[1] / image.shape[0]
    height = int(width / aspect_ratio)
    return cv.resize(image, (width, height))

def findObjects(results, img, classNames, frame_number, fps, save_images=False):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for result in results:
        for det in result.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            bbox.append([x1, y1, x2 - x1, y2 - y1])
            classIds.append(int(det.cls[0]))
            confs.append(float(det.conf[0]))

    # Perform non-maximum suppression
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                       (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if save_images and classNames == classNames1:
                # Calculate timestamp
                seconds = frame_number / fps
                timestamp = str(datetime.timedelta(seconds=seconds)).split(".")[0]

                # Save image to REPORTE folder
                img_name = f'{classNames[classIds[i]]}_{int(confs[i] * 100)}%_{timestamp}.png'
                img_path = os.path.join('REPORTE', img_name)
                cv.imwrite(img_path, img)

# Alternate between models
use_model1 = True
fps = cap.get(cv.CAP_PROP_FPS)
frame_number = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_number += 1

    # Switch between models
    if use_model1:
        results = model1(img)
        classNames = classNames1
        save_images = True
    else:
        results = model2(img)
        classNames = classNames2
        save_images = False
    
    use_model1 = not use_model1  # Toggle model

    # Find objects
    findObjects(results, img, classNames, frame_number, fps, save_images=save_images)

    # Resize the image to fit the target size maintaining aspect ratio
    img_resized = resize_with_aspect_ratio(img, target_width)

    # Check if the resized image is larger than the target height and adjust if necessary
    if img_resized.shape[0] > target_height:
        scale_factor = target_height / img_resized.shape[0]
        img_resized = cv.resize(img_resized, None, fx=scale_factor, fy=scale_factor)
    
    # Display the image
    cv.imshow(window_name, img_resized)

    # Calculate the position to center the window
    window_x = (screen_width - img_resized.shape[1]) // 2
    window_y = (screen_height - img_resized.shape[0]) // 2

    # Move the window to the center of the screen
    cv.moveWindow(window_name, window_x, window_y)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

