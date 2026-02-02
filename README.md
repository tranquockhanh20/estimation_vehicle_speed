Vehicle Speed Estimation Using Computer Vision
Overview

This project focuses on applying computer vision and deep learning techniques to estimate the moving speed of road vehicles using video data from a static camera.
Instead of relying on traditional physical sensors (radar, lidar), the system uses image-based analysis to provide a low-cost, flexible, and scalable solution for intelligent traffic monitoring.

Main Objectives

Detect vehicles from video streams using a deep learning model.

Track each vehicle consistently across frames.

Estimate vehicle speed in real-world units (km/h) based on visual motion.

Operate in near real-time on a personal computer.

Methodology

The system consists of the following main stages:

Vehicle Detection:
YOLOv8 is used to detect vehicles in each video frame.

Multi-Object Tracking:
ByteTrack assigns and maintains a unique ID for each vehicle during movement.

Perspective Transformation:
A homography-based perspective transform maps pixel coordinates to real-world distances (meters).

Speed Estimation:
Vehicle speed is computed using displacement over time:

𝑣
=
𝑠
𝑡
v=
t
s
	​


and converted to km/h.

Features

Real-time vehicle detection and tracking

Speed estimation for multiple vehicles simultaneously

On-screen display of bounding boxes, IDs, and speed values

Automatic image capture for speeding vehicles

Works with both recorded videos and live camera streams

Experimental Results

Stable performance at 25–30 FPS with GPU support

Reliable ID tracking without frequent identity switching

Accurate speed estimation under proper camera calibration

Suitable for intelligent traffic surveillance applications

Technologies Used

Python

PyTorch

Ultralytics YOLOv8

ByteTrack

OpenCV

Limitations

Accuracy depends on camera angle and perspective calibration

Performance may degrade in poor lighting or adverse weather

Best suited for vehicles moving along a dominant direction

Future Work

Improve robustness under challenging environmental conditions

Integrate license plate recognition

Extend to full Intelligent Transportation Systems (ITS)
