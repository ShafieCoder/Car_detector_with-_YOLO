# YOLO_algorithm_detector
You are working on a self-driving car. Go you! As a critical component of this project, you'd like to first build a car detection system. The first step is collecting data:
for this goal, you've mounted a camera to the hood of the car, which takes pictures of the road ahead every few seconds as you drive around. You've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found.

##  YOLO
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

## Model Details
#### Inputs and outputs
The input is a batch of images, and each image has the shape (m, 608, 608, 3)
The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  <img src="https://render.githubusercontent.com/render/math?math=(p_c,b_x,b_y,b_h,b_w,c)">. If you expand  <img src="https://render.githubusercontent.com/render/math?math=c">  into an 80-dimensional vector, each bounding box is then represented by 85 numbers.
