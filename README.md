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
 We're trying to detect 80 classes, and are using 5 anchor boxes. The information on the 80 classes and 5 boxes is gathered in two files: "coco_classes.txt" and "yolo_anchors.txt". We'll read class names and anchors from text files. The car detection dataset has 720x1280 images, which are pre-processed into 608x608 images.
 
 #### 2.  Loading a Pre-trained Model
 Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. We should load an existing pre-trained Keras YOLO model. The weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of the Readme. Technically, these are the parameters from the "YOLOv2" model, but are simply referred to as "YOLO" in this Readme.
 
 __Reminder__: This model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85).
 
 #### 3. Convert Output of the Model to Usable Bounding Box Tensors
 The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. We will need to call yolo_head to format the encoding of the model we got from yolo_model into something decipherable:
 
 yolo_model_outputs = yolo_model(image_data) yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names)) The variable yolo_outputs will be defined as a set of 4 tensors that we can then use as input by our yolo_eval function.
 
 #### 4. Filtering Boxes
 yolo_outputs gave us all the predicted boxes of yolo_model in the correct format. To perform filtering and select only the best boxes, we will call yolo_eval, which we had previously implemented, to do so:
 
 out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)
 
 #### 5. Run the YOLO on an Image
 Let the fun begin! We will create a graph that can be summarized as follows:
 
 yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output yolo_model.output is processed by yolo_head. It gives us yolo_outputs yolo_outputs goes through a filtering function, yolo_eval. It outputs our predictions: out_scores, out_boxes, out_classes.
 
 Now, we have implemented the predict(image_file) function, which runs the graph to test YOLO on an image to compute out_scores, out_boxes, out_classes.
 
 The code below also uses the following function:
 
 image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
 
 which opens the image file and scales, reshapes and normalizes the image. It returns the outputs:
 
image: a python (PIL) representation of our image used for drawing boxes. We won't need to use it.
image_data: a numpy-array representing the image. This will be the input to the CNN.

## Summary for YOLO
* Input image (608, 608, 3)
* The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
* After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
  * Each cell in a 19x19 grid over the input image gives 425 numbers.
  * 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
  * 85 = 5 + 80 where 5 is because  <img src="https://render.githubusercontent.com/render/math?math=(p_c,b_x,b_y,b_h,b_w)"> has 5 numbers, and 80 is the number of classes we'd like to detect.
* You then select only few boxes based on:
  * Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
  * Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
* This gives us YOLO's final output.

## References
The ideas presented here came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository and Convolutional Neural Network course taught by Andrew Ng in Coursera. The pre-trained weights used in this exercise came from the official YOLO website.

* Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
* Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
* Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
* The official YOLO website [(https://pjreddie.com/darknet/yolo/)](https://pjreddie.com/darknet/yolo/)
