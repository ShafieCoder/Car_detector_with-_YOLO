# YOLO_algorithm
We are working on a self-driving car. As a critical component of this project, we'd like to first build a car detection system. The first step is collecting data:
for this goal, we've mounted a camera to the hood of the car, which takes pictures of the road ahead every few seconds as we drive around. we've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found.

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

•	To write Math: 
<img src="https://render.githubusercontent.com/render/math?math=(m,n_H,n_W,anchors,classes)">


If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425). See in the following Figure.
![alt text](https://github.com/ShafieCoder/Car_detector_with-_YOLO/blob/main/Images_folder/flatten.png?raw=true)

#### Class score
Now, for each box (of each cell) we'll compute the following element-wise product and extract a probability that the box contains a certain class. 

The class score is <img src="https://render.githubusercontent.com/render/math?math=score_{c,i}=p_c \times c_i"> : the probability that there is an object <img src="https://render.githubusercontent.com/render/math?math=p_c "> times the probability that the object is a certain class <img src="https://render.githubusercontent.com/render/math?math=c_i">.

#### Non-Max suppression
Even if we consider only those bounding boxes for which the model had assigned a high probablity, but this is still too many boxes. We'd like to reduce the algorithm's output to a much smaller number of detected objects. 

To do so, we'll use __non-max supperssion__.  Specifically, we'll carry out these steps:
* Get rid of boxes with a low score.  Meaning, the box is not very confident about detecting a class, either due to the low probability of any object, or low probability of this particular class.
* Select only one box when several boxes overlap with each other and detect the same object.

## Details of implementing YOLO algorithms
#### 1. Filtering with Threshold on Class Scores
You're going to first apply a filter by thresholding, meaning you'll get rid of any box for which the class "score" is less than a chosen threshold.

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It's convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:
* box_confidence: tensor of shape  (19,19,5,1) containing  pcpc  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
*boxes: tensor of shape  (19,19,5,4)  containing the midpoint and dimensions  <img src="https://render.githubusercontent.com/render/math?math=(b_x,b_y,b_h,b_w)">  for each of the 5 boxes in each cell.
* box_class_probs: tensor of shape  (19,19,5,80)  containing the "class probabilities"  <img src="https://render.githubusercontent.com/render/math?math=(c_1,c_2,...c_80)">  for each of the 80 classes for each of the 5 boxes per cell.

In the first step we implement yolo_filter_boxes(). To implement this function, we should follow the following steps:
1. Compute box scores by doing the elementwise product
2. For each box, find:
  * the index of the class with the maximum box score
  * the corresponding box score

  __Useful References__
  - [tf.math.argmax](https://www.tensorflow.org/api_docs/python/tf/math/argmax)
  - [tf.math.reduce_max](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max)
  - [tf.boolean_mask](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)
  
  __Helpful Hints__
 * For the axis parameter of argmax and reduce_max, if you want to select the last axis, one way to do so is to set axis=-1. This is similar to Python array indexing, where you can select the last position of an array using arrayname[-1].
 * Applying reduce_max normally collapses the axis for which the maximum is applied. keepdims=False is the default option, and allows that dimension to be removed. You don't need to keep the last dimension after applying the maximum here.
 
 3. Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be True for the boxes you want to keep.
 4. Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes you don't want. You should be left with just the subset of boxes you want to keep.
 
 #### 2. Non-max Suppression
 
  Even after filtering by thresholding over the class scores, you still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS).
  
  Non-max suppression uses the very important function called "Intersection over Union", or IoU.
  To implement iou(), use the following hints:
  
  * we use the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) is the lower-right corner. In other words, the (0,0) origin starts at the top left corner of the image. As x increases, you move to the right. As y increases, you move down.
  * To implement iou() function, a box is defined using its two corners: upper left <img src="https://render.githubusercontent.com/render/math?math=(x_1,y_1)">   and lower right  <img src="https://render.githubusercontent.com/render/math?math=(x_2,y_2)"> , instead of using the midpoint, height and width. This makes it a bit easier to calculate the intersection.
  * To calculate the area of a rectangle, multiply its height  <img src="https://render.githubusercontent.com/render/math?math=(y_2-y_1)">  by its width  <img src="https://render.githubusercontent.com/render/math?math=(x_2-x_1)"> . Since  <img src="https://render.githubusercontent.com/render/math?math=(x_1,y_1)">  is the top left and  <img src="https://render.githubusercontent.com/render/math?math=(x_2,y_2)">  are the bottom right, these differences should be non-negative.
  * The intersection of the two boxes is  <img src="https://render.githubusercontent.com/render/math?math=(xi_1,yi_1,xi_2,yi_2)">:
    * xi_1 = maximum of the x1 coordinates of the two boxes
    * yi_1 = maximum of the y1 coordinates of the two boxes
    * xi_2 = minimum of the x2 coordinates of the two boxes
    * yi_2 = minimum of the y2 coordinates of the two boxes
    * inter_area =  You can use max(height, 0) and max(width, 0)

#### 3. YOLO Non-max Suppression
We are now ready to implement non-max suppression. The key steps are:

1. Select the box that has the highest score.
2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= iou_threshold).
3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.
We Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so we don't actually need to use our iou() implementation):
* [tf.image.non_max_suppression()] (https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
* [tf.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)
  
  #### 4. Wrapping Up the Filtering
 It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented. 
 
 Implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions :
 
 boxes = yolo_boxes_to_corners(box_xy, box_wh)
 
 which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates <img src="https://render.githubusercontent.com/render/math?math=(x_1, y_1, x_2, y_2)"> to fit the input of yolo_filter_boxes.
 
 boxes = scale_boxes(boxes, image_shape)
 
 YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image -- for example, the car detection dataset had 720x1280 images -- this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.
 
 ## Test YOLO Pre-trained Model on Images 
 Now we are going to use a pre-trained model and test it on the car detection dataset. For this goal, we do following steps:
 
 #### 1.  Defining Classes, Anchors and Image Shape
 We're trying to detect 80 classes, and are using 5 anchor boxes. The information on the 80 classes and 5 boxes is gathered in two files: "coco_classes.txt" and "yolo_anchors.txt". You'll read class names and anchors from text files. The car detection dataset has 720x1280 images, which are pre-processed into 608x608 images.
 
 

