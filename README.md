# YOLO_algorithm
You are working on a self-driving car. Go you! As a critical component of this project, you'd like to first build a car detection system. The first step is collecting data:
for this goal, you've mounted a camera to the hood of the car, which takes pictures of the road ahead every few seconds as you drive around. You've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found.

##  YOLO
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

## Model Details
#### Inputs and outputs
* The input is a batch of images, and each image has the shape (m, 608, 608, 3)
* The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  <img src="https://render.githubusercontent.com/render/math?math=(p_c,b_x,b_y,b_h,b_w,c)">. If you expand  <img src="https://render.githubusercontent.com/render/math?math=c">  into an 80-dimensional vector, each bounding box is then represented by 85 numbers.

#### Anchor Boxes
* Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes. In this project, 5 anchor boxes were chosen (to cover the 80 classes), and stored in the file "./model_data/yolo_anchors.txt"
* The dimension for anchor boxes is the second to last dimension in the encoding: <img src="https://render.githubusercontent.com/render/math?math=(m,n_H,n_W,anchors,classes)"> .
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

#### Encoding
Let's look in greater detail at what this encoding represents.

![alt text](https://github.com/ShafieCoder/Car_detector_with-_YOLO/blob/main/Images_folder/encoding.png?raw=true)

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425). See in the following Figure.
![alt text](https://github.com/ShafieCoder/Car_detector_with-_YOLO/blob/main/Images_folder/flatten.png?raw=true)
