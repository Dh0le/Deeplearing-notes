# Object Detection

## Object localization

There are three steps of object detection problem:

- Object Classification: Check if there is target object in image
- Object Localization: locate target object
- Object detection: check different object and locate them.

![](img\Obj detection.png)

Example:

Assume we have traditional Conv network for classification that output through softmax with classification of pedestrian, car, motorcycle and nothing.

We can turn this network to output the location of the object. We can output four more numbers like $b_x,b_y,b_h,b_w$ . Based on these numbers we can create a bounding box. 

Overall, we need to output y that contains $[p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3]$

$p_c$ means that if there are target objects in the picture. 

This is important because if $p_c=0$ then the rest of other parameters mean nothing.

![](img\Objdetection1.png)

- Loss function: Square error

  - $(\hat y-y_1)^2+(\hat y-y_2)^2....+(\hat y-y_8)^2 $if $p_c=1$
  - $(\hat y-y_1)^2$ if $p_c = 0$ since rest parameter we don't care.

  We can used different loss function over different output parameters.

![](img\ObjDetection2.png)



## Landmark detection

According to the previous object detection problem, we can output landmark over a picture to achieve feature detection and localization.

We can output some landmark like eye corner in facial recognition, skeleton points in human pose recognition. It is simple that we only need to add output unit to do that. But please remember that the order of landmark output must stay the same.

![](img\landmark.png)

## Object detection using sliding window

1. We can train a classifier using a Conv Net and we use cropped image that Object occupied a big portion to train it. ![](img\SlidWindow.png)
2. Then we use a sliding window on the image we want to process and then input the image in the window into the Conv net.
3. We can use different size of windows for the Conv net.

![](img\SlidingWindow2.png)

The computational cost of it method is pretty high.

- Using big stride will lower the cost but it will affect the performance.

## Convolutional implementation 

### Turning FC layer into conv layers

As we previously mentioned, for FC layer in CNN, we can use 1 x 1 conv layer to replace fully connected layer.![](img\FCtoConv.png)

## Sliding window implementation:

![](img\ConvImplementation.png)

Based on the model we trained  above on the image. When we input an 16x16x3 image into the network.  And used blue color to mark the sliding window. When we take 2 cell as stride, at the end, we will have 4 image with 10x10x16. Since there are many over lap area in each sliding window, there are a lot of redundant calculation.  The convolutional implementation will help us with this problem. 

The operation of a sliding window is basically the same as a convolution operation. So we can direction pass the entire image into the network We will have 12x12x16 for the next layers. If we do the conv implementation for the following layer, we can use conv layer to share the calculation of the overlap area.



![](img\ConvImplementation2.png)

Based on the method we have above, we can input the entire image into the network and then we only need one forward propagation and we will have the prediction to every subset.

## Bounding box

After we use convolution implementation on sliding window, the efficiency of calculation was successfully improved.  But there is still a big problem: we cannot output the precise window that contains the target object. It is highly possible that one of the sliding window might contain the most part of target object but not all of it. 



### YOLO algorithm

YOLO algorithm makes sliding window find more accurate box.

1. We sperate the image into n x n grid cells
2. Run image detection and localization on each gird cells
3. Define training labels for every grid cell. In this example we will just follow the previous $y_1 = [p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3]$, so that for each grid cell i we have $y_i$
4. Combine n x n grid cells together and at the end the output n x n x 8 (Because we have 8 features we want)

After we trained the model, when we input image with the same size we can get the n x n x 8 output. By observe the output we can tell that if there is a object in the grid cell and output the precise boundary.

**YOLO notation:**

- The YOLO algorithm will take a look at the mid point of every located object and assign that object to the grid cell that has the mid point.
- YOLO output the boundary explicitly so that we can have any height and width ratio and output more precise coordinate without limitation of stride.
- YOLO is a one time convolution operation which can quick output what we want.

### Bounding box detail

The parameter in the training label affect the precision greatly.  Here is a reasonable set of parameters

- For each grid cell, the top left corner was marked as (0,0), the bottom right corner was marked as (1,1)
- $b_x,b_y$ are the coordinate numbers that stays between (0,1)
- $b_h,b_w$ are ration that of height and width to compare to current grid and they could be greater than 1.

## Intersection over union

Evaluating object localization:

Intersection over union（IoU): Is the ratio of region that overlap by ground true region + output region(Green) and their intersection(Yellow) in following image.

![](img\IoU.png)

IoU = size of (yellow region)/ size of (green region)

Normally, the result was considered "Correct"  if IoU is greater or equal to 0.5

You can also set higher threshold value.

This a great to judge if the output is good enough.





## Non-max suppression

Non max suppression is a method that make sure you algorithm only detect each object once.

Lets  take a look at an example:

- We are looking for car, pedestrian and bike in following image

- We put a 19 x 19 grid cells

- In this case many cells might detect cars(bunch of green and yellow)

  Non- max suppression will help the algorithm get rid of the other except for the one with center of object.

![](img\NoneMAXs0.png)



- As we can see, there are many blue window that shows they detect the object
- Most of our operation will be based on two values($p_c$ and IoU)
- We use a threshold value like $p_c=0.6$ to get rid of some unconfident windows
- Then we pick the most confident window and use it as the output.
- Get rid of windows that have $IoU>0.5$ with output grid.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\Coursera-deeplearning-Note\img\NoneMax1.png)

If we have multiple class, then we should run this Non-max suppression on each class.

![](img\NoneMaxs2.png)



## Anchor Box

 Based on the algorithm we have above, we can only detect one object in one grid cell. But there is case that multiple object could be showed up in one grid cell.  By using anchor box, we can handle this situation. 

### Overlap object:

For overlap objects, some of their central points may be assigned to same grid cell. The solution for this situation is to predefine several anchor boxes and link all the predefine boxes into 1 output:

$y_i=[p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3,p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3....]$

- Without anchor box: for each object, we assign it into grid cell that contain their central points like(n x n x 8)
- With anchor box: For each training object we not only assign it to the grid but also assign it with a anchor box that has highest IoU. (Like n x n x 16 with two anchor box)

![](img\AnchorBox0.png)



If there is only 1 object in the grid cell, we kind of don't care the parameters following the $p_c =0$

**Problem cannot be gracefully handled.**

- If the object number is greater than the number of anchor boxes, we need some other method to handle that.
- If there are two object with same shape of the anchor box, we need some other method to handle that.

### How to choose anchor box shape:

- Normally we assign the shape to anchor box. 5-10 shapes will basically handle our needs.
- Using K-means algorithm: To do clustering on several shapes and use the result to represent a type of anchor box.



## YOLO detection

In this chapter, we will put every thing before together.

For training set we define classes and training labels.

The size of the output $y = n\times n\times anchor\ box\times(5+num\ classes)$

$n$ is the number of gird you want to use.

The neural network will just go through the image and and output this y vector that gives you object class and bounding box. 

![](img\YOLO1.png)

![](img\YOLO2.png)

At the end, we run non max suppression. 

For example we have 2 anchor boxes:

-  For each grid cells we get 2 predicted bounding boxes.

- Get rid of low probability predictions.
- For each class use non-max suppression to generate final predictions.

![](img\YOLO3.png)



## RPN NETWORK

Region proposal :R-CNN

RPN dose not go through all the windows, instead if pick some region that has high probability to contain target object.  So it save computational power.

![](img\RCNN0.png)

Implementation:

- Run segmentation algorithm which separate image into several colored block and place the window on that block.
- Input the content on that block into the network.

#### Faster algorithm

R-CNN: Propose regions,Classify  proposed region one at a time. Output label and bounding box.

Fast R-CNN: Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.

Faster R-CNN : Use convolutional network to propose regions.



![](img\RCNN1.png)







