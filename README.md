# Vehicle Speed Estimation Using Computer Vision

<img width="1198" height="565" alt="image" src="https://github.com/user-attachments/assets/2f1e2c47-e0b1-4b4d-86f4-3d92a6a1444a" />

## Overview

This project explores the application of **computer vision and deep learning** to estimate the **speed of moving vehicles** using video data captured from a static camera.

The system is capable of detecting, tracking, and estimating vehicle speed from video streams. It is designed to operate in **near real-time** and provides visual outputs that support traffic monitoring and basic speed violation alerting.

---

## Project Motivation

Traditional vehicle speed measurement methods, such as radar or physical sensors, are accurate but often costly and difficult to deploy at scale.

This project investigates whether **camera-based solutions** using deep learning can serve as a more flexible and cost-effective alternative for traffic analysis and monitoring.

---

## System Pipeline

The system follows a standard computer vision workflow:

1. **Vehicle Detection**
   Vehicles are detected in each frame using the **YOLOv8** object detection model.

2. **Multi-Object Tracking**
   The **ByteTrack** algorithm is used to assign and maintain consistent IDs for vehicles across consecutive frames.

3. **Perspective Transformation**
   A manually calibrated perspective transformation maps image coordinates (pixels) to approximate real-world distances (meters).

4. **Speed Estimation**
   Vehicle speed is estimated by measuring the displacement of tracked vehicles over time and converting it to kilometers per hour (km/h).

---

## Features

* Detects multiple vehicles in video streams
* Tracks vehicles using unique IDs
* Estimates and displays vehicle speed in real time
* Supports both recorded video files and live camera input
* Automatically saves images of vehicles exceeding a predefined speed threshold

---

## Technologies Used

* **Python**
* **PyTorch**
* **YOLOv8 (Ultralytics)**
* **ByteTrack**
* **OpenCV**

---

## Results

* The system runs at approximately **25–30 FPS** on a GPU-enabled personal computer
* Vehicle tracking remains stable under normal conditions
* Speed estimation is reasonably accurate after proper camera calibration
