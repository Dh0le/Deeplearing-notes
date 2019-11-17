# CNN

## Computer vision

Typical computer vision problem:

1. Image classification
2. Object detection
3. Neural Style Transfer
   - Transfer a real life image into a Picasso's style painting image. 

## Edge detection

Understand the mechanism of filter:

![](img\filter.png)

Using a filter to get the vertical edges:

![](img\vertical edge.png)

## Other edge detection

Different filter will help developer to detect different edges.



Sobel filter: put little bit more weight into central pixel thus make the result more robust.

Scharr filter: Vertical edge detection filter

Make the filter number as parameters to learn instead of hand coded filter.

use backpropagation to compute these number.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\filter1.png)



### Padding

If we have an **n x n** matrix, and a **f x f** filter then the output filter size will **n - f +1 x  n - f + 1**. 

This will have several downsides:

1. We can tell that the image starts to shrink. 
2. The pixels in the corner region are used much less in the output which means we lost the information on the corner area.

The solution for these problem is padding which is to pad the image before convolute operation.

By padding 6 x 6 image into 8 x 8 image, we can still get 6 x 6 image after convolution operation. 

This this case out padding size is p = 1which means we pad 1 pixel into every edge pixel.

then the output image size will be **n + 2p - f  + 1 x n + 2p - f  + 1 **. 

By doing that the problem 2 were reduce. 

Two common choice for padding

1. "Valid" which mean no padding. $n - f +1 x  n - f + 1$

2. "Same" Pad so that output size is the same as the input size. $n + 2p - f  + 1 x n + 2p - f  + 1$ which means $p = \frac{f-1}{2}$

f is usual odd by convention

1. if f is even then you need asymmetric padding.
2. Odd filter will give you central.



## Strided convolutions

Lets take the stride into account, then the equation becomes:

$(\frac{n+2p-f}{s}+1) \times(\frac{n+2p-f}{s}+1)$

It does have possibilities that $(\frac{n+2p-f}{s}+1) $ is not an integer. If we encounter this situation, by convention we should round down the $(\frac{n+2p-f}{s}+1) $ which means you filter must be entirely filled to start the computation. 

![](img\convlution_equa1.png)

#### Technical note on cross-correlation vs. convolution

We actually doing **cross-correlation** in the the deep learning.  Since by mathematic definition, we need to flip the filter before we start to do the  convolution operation. In deep learning, omitting this double mirroring operation just simplify the code and still works well. But we do call this operation **convolution** in deep learning.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\math_convolu.png)

## Convolution over volumes



To start the convolution over the volume, we introduce the concept of channels like RGB channels. So that the 2D image becomes 3D volumes. 

In order to do the convolution operation, we also make filter a volume instead of a image. 

By convention the number of channel for filter stays identical with the the number of channels in image. 

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\multiple_filters.png)

If we want to detect different feature through different filters. We can stack up the result of different filter so it makes the output also a volume.

Summary:

When we have a $n\times n\times n_c$ image. ($n_c$ is the number of channels) and we a filter with size of $f \times f \times n_c$ the output of the convolutional operation will have size $n-f+1 \times n-f+1\times n_c^{'}$  and $n_c^{'}$ is the number of filters you chose to used.



## Convolutional layer

The matrix compute by the filter played the role as $W^{[l]}a^{[l-1]}$

Then we add bias elementwise to matrix then the matrix becomes $Z^{[l]} = W^{[l]}a^{[l-1]}+b^{l}$

then we apply the non-linear activation function elementwise to the matrix, that is the process we have from $a^{[l-1]}$ to $a^{[l]}$

#### Summary of notation:

Layer $l$ is a convolution layers:

$f^{[l]} = filter \ size$

$p^{[l]}= padding$

$s^{[l]}=stride$

$n^{[l]}_c = number\  of \ filters$

$n^{[l]}_W= \frac{n^{[l-1]}_W+2p^{[l]}-f}{s^{[l]}}+1$ and remember to round down if it is not an integer.

$n^{[l]}_H = \frac{n^{[l-1]}_H+2p^{[l]}-f}{s^{[l]}}+1$

The input size will be $n^{l-1}_H \times n^{[l-1]}_W \times n_c^{[l-1]}$

The output size will be $n_H^{[l]}\times n^{[l]}_W\times n_c^{[l]}$

Each filter is : $f^{[l]}\times f^{[l]} \times n^{[l]}_c$

Activations: $a^{[l]} = n_H^{[l]}\times n^{[l]}_W\times n_c^{[l]}$

Activation output: $A^{[l]}=m\times n^{[l]}_H \times n^{[l]}_W \times n_c^{[l]}$ which $m$ is the number of examples.

Weight: $f^{[l]}\times f^{[l]}\times n^{[l]}_c \times n^{[l]}_c$

Bias: $1\times 1\times 1\times n^{[l]}_c$



Type of layers in a convolutional network:

- Convolution: (CONV)
- Pooling:(POOL)
- Fully connected (FC)



## Pooling layer:

### Max pooling: 

Max pooling layer is to to keep the max value in a region and store them into a relatively smaller matrix. This means that if we detect a feature in an area and we keep its max value this value might be applied to other area.

The pooling operation is a way that can lower the dimensional while still keep feature. Here is the picture that might help you to understand the work done by max pooling.

![](img\maxpooling.png)



And the max pooling dose have two hyperparameters $f$ (filter size)and $s$(stride) but there is nothing for gradient descent to learn, once these hyperparameter were decided, it just a fixed computation. The equation we used  to compute the output of the  max pooling is the same as we do for convolution layer.

And max pooling operation for each channel works independently , so that the output of max pooling will also have the same channel numbers. 

### Average Pooling:

This average pooling is less often used as max pooling. Instead of using the max value in an area, we calculate the average and output that value.

![](img\averagepooling.png)



I will list some examples that fits human intuition to help readers to understand this pooling operation.

To introduce these example we have to know the concept of invariance which includes **translation, rotation and scale**

1. Translation which means you move you target object in the image vertically or horizontally. Following picture will show you why the max pooling is a great way to deal with this type of moving.![](img\maxpooling_trans.jpg)

2. Rotation invariance, rotation is somehow not effective for some letters in some languages or mix used of some languages, since the rotations may somehow change the meaning of it. But other than that it is okay. ![](img\maxpooling_rotate.jpg)

3. Scale invariance is that the meaning of a certain features might not change with its scale. Let's use 0 in pixel as an example:

   ![](img\maxpooling_scale.jpg)



## Neural network example



This example is inspired by **LeNet-5**

The following picture will show the computation graph of this example.

![](img\LeNet-5.png)

And here is the chart that gives you all the activation shape and sizes and number of parameters in previous model:

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\LeNet-5_chart.png)

**Notation：**

Some of you might wonder how to calcualte the number of parameters in the chart above. This **parameters** is that the number of values your model can learn. So it includes both **weight and bias**. 

And remember the fact that the filter **share same weight and bias in the same channel**. So the equation to calculate the number of parameters is:

$paranum = (f_w\times f_h+1(bias))\times n_c$

In some definition of **layers**, only layers with weight can be ascribed as a layer. So people sometimes will put conv and pooling into 1 layer instead of 2.

**Notation **: It is sometimes good to **not** set hyperparameters yourself, but instead go to others paper and try to used their set of hyperparameters.  So **Go** see others work and gain experience of how other people put these parts together.



## Why convolutions

1. Parameter sharing : A feature detector(such as a vertical edge detector) that 's useful in one part of the image is probably useful in another part of image. This is a great way to effectively reduce the number of parameters and it works both for low level feature(edges) and high level feature(eyes, nose). 
2. Sparsity of connections: In each layer, each output value depends only on a small number of inputs. This reduction of parameters  allows developers to train the network with smaller training set and prevent overfitting.
3. Robust to invariance: we have discuess in previous chapter.

