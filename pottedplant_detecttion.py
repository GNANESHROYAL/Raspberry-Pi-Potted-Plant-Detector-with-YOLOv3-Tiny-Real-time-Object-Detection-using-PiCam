#    ----------    Author    ----------
# Name      : Gnanesh.k
# Contact   : gnaneshroyal254@gmail.com
# Linkedin  : https://www.linkedin.com/in/gnanesh-royal-374126213
# Github    : https://github.com/GNANESHROYAL/ROS-ROBOTICS.git

import cv2 as cv
import time
import numpy as np

# Thresholds for confidence and Non-Maximum Suppression (NMS)
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# Colors for drawing bounding boxes around detected objects
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Font for labeling detected objects
fonts = cv.FONT_HERSHEY_COMPLEX

# Load the class names used by the YOLOv3 Tiny model
class_names = []

# Change the address according to your location: "/home/ubuntu/yolov4/final/darknet/data/coco.names"
with open("/home/ubuntu/yolov4/final/darknet/data/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the YOLOv3 Tiny model using OpenCV's deep neural network module
yoloNet = cv.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg','DNN_TARGET_MYRIAD')

# Set preferable backend and target for the model to run efficiently on GPU (if available)
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# Create a detection model from the YOLOv3 Tiny network
model = cv.dnn_DetectionModel(yoloNet)

# Set input parameters for the model
model.setInputParams(size=(416, 416), scale=0.00392, swapRB=True)

# Function to detect potted plants in an image
def ObjectDetector(image):
    # Use the YOLOv3 Tiny model to detect objects in the image
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    coordinates = []
    
    # Loop through the detected objects
    for (classid, score, box) in zip(classes, scores, boxes):
        # Check if the detected object is a "pottedplant"
        if class_names[classid[0]] == 'pottedplant':
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            
            # Draw a rectangle around the detected object
            cv.rectangle(image, box, color, 2)
            
            # Put the label with the confidence score on top of the rectangle
            cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)
            
            # Determine the position of the center of the detected object in the frame
            object_center_x = (box[0] + box[2] // 2)
            frame_part = image.shape[1] // 7  # Divide the frame width into 7 parts
            
            # Classify the position of the object relative to the frame center
            if object_center_x < frame_part:
                position_output = "-3"  # Object is at the far left (-3)
            elif object_center_x < frame_part * 2:
                position_output = "-2"  # Object is at the left (-2)
            elif object_center_x < frame_part * 3:
                position_output = "-1"  # Object is slightly left (-1)
            elif object_center_x > frame_part * 6:
                position_output = "+3"  # Object is at the far right (+3)
            elif object_center_x > frame_part * 5:
                position_output = "+2"  # Object is at the right (+2)
            elif object_center_x > frame_part * 4:
                position_output = "+1"  # Object is slightly right (+1)
            else:
                position_output = "0"   # Object is centered (0)
            print(position_output)
                        
            coordinates.append(box)  # Append the bounding box coordinates to the list
    return coordinates

# Open the PiCam for video streaming
camera = cv.VideoCapture(0)

counter = 0

# Start processing frames from the camera stream
while True:
    ret, frame = camera.read()
    counter += 1
    
    # Skip every other frame to reduce processing load
    if counter % 2 != 0:
        continue
    
    # Break the loop if there is no more frame or user presses 'q'
    if not ret or cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Detect potted plants in the frame using the ObjectDetector function
    object_coordinates = ObjectDetector(frame)
    
    # Display the frame with detected objects
    cv.imshow('YOLOv3 Tiny Potted Plant Detection', frame)

# Release the camera and close the window when done
camera.release()
cv.destroyAllWindows()
