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