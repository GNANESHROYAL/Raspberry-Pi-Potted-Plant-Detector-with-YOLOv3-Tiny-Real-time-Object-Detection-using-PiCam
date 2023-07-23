# Raspberry-Pi-Potted-Plant-Detector-with-YOLOv3-Tiny-Real-time-Object-Detection-using-PiCam
"Real-time potted plant detection using YOLOv3-Tiny optimized for Raspberry Pi with PiCam. Efficient and easy-to-use object detection project."
# Raspberry Pi Potted Plant Detector

![final](https://github.com/GNANESHROYAL/Raspberry-Pi-Potted-Plant-Detector-with-YOLOv3-Tiny-Real-time-Object-Detection-using-PiCam/assets/113758576/93663876-6eaa-4f05-8c0a-a71f7a23e603)

# Potted Plant Detection using YOLOv3-tiny


## Overview

This repository contains a Python script that utilizes YOLOv3-tiny, a lightweight version of the YOLO (You Only Look Once) object detection model, to detect potted plants in real-time using a Raspberry Pi and a Picam. The script captures frames from the Picam, performs object detection, and displays the results with bounding boxes around the detected potted plants. Additionally, it prints the position (PID values) of the detected potted plants relative to the center of the frame.

## Requirements

- Raspberry Pi (tested on Raspberry Pi 3 Model B)
- Raspberry Pi camera (Picam) or a compatible webcam
- Python 3.x
- OpenCV (Python bindings)

## Installation

1. Clone this repository to your Raspberry Pi:

```bash
git clone https://github.com/GNANESHROYAL/Raspberry-Pi-Potted-Plant-Detector-with-YOLOv3-Tiny-Real-time-Object-Detection-using-PiCam.git
```

2. Navigate to the cloned repository directory:

```bash
cd potted-plant-detection
```

3. Install the required Python packages:

```bash
pip install opencv-python
```

4. Download the YOLOv3-tiny weights and configuration files from the official website or another trusted source and place them in the repository directory. Make sure the files are named as follows:

```
yolov3-tiny.weights
yolov3-tiny.cfg
```

## Usage

1. Connect the Picam to the Raspberry Pi if not already connected.

2. Run the Python script:

```bash
python potted_plant_detection.py
```

3. The script will open a window showing the live camera feed with detected potted plants highlighted by bounding boxes and labels. The terminal will print the PID values of the detected potted plants relative to the center of the frame.

4. To exit the script, press the 'q' key.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The YOLOv3-tiny model used in this project is based on the original YOLOv3 model by Joseph Redmon and the YOLO authors. For more information, visit: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/).

- The COCO dataset is used for training the YOLO model. For more information, visit: [COCO Dataset](https://cocodataset.org/).

## Tags

`Raspberry Pi`
`Object Detection`
`YOLOv3`
`YOLOv3-tiny`
`Computer Vision`
`Deep Learning`
`Convolutional Neural Networks`
`Picam`
`Real-time Detection`
`OpenCV`
`Python`
`Robotics`
`IoT`
`Edge Computing`
`Machine Learning`
`COCO Dataset`
`Open Source`

## Disclaimer

This project is provided as-is and is intended for educational purposes only. The authors are not responsible for any misuse or damage caused by using this code.
