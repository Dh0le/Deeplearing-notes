# Face Recognition

## What is face recognition

### Face verification vs. Face recognition

Verification:

- Input image, name/ID
- Output whether the input image is that of the claimed person

Recognition

- Has a database of K person
- Get an input image
- Output ID if the image is any of the K persons (Or "Not recognized")

The accuracy of verification needs to be very good(like 99.9999%)  so it can used in Recognition system.

## One shot Learning

We need to recognize the identity of a person by only one image. In another word, we need to develop a model that learning from one example to recognize the person again.

Learning a "similarity" function

d(img1,img2) = degree of difference between images.

If $d(img1,img2)<=\tau$  Then predict same.

if  $d(img1,img2)>\tau$  Then predict different.

As long as we can output d(image1,image2) we can solve this one shot problem.


## Siamese network

Create a network that output a vector(like 128d vector) based on the input of an image.  This output vector $f(x^{(1)})$ is the encoding representation of this image. 

Then we do the same for another image and get $f(x^{(2)})$  

Then we can calculate the difference between these two vector representation of each image with : $diff = ||f(x^{(i)}-f(x^{(j)}))||$

If $x^{(i)}$ and $x^{(j)}$ are images of a same person, the diff should be very small.

If $x^{(i)}$ and $x^{(j)}$ are images of a different person, the diff should be very large.

 

## Triplet loss

### Learning Objective

Because we are always looking at three images (Anchor, Positive(Same with anchor), Negative(Different form anchor))

What we want is $\frac{||f(A)-f(P)||^2}{distance(A,P)}<= \frac{||f(A)-f(N))||}{distance(A,N)}$

which can be written as $||f(A)-f(P)||^2-||{f(A)-f(N)}||^2<=0$ But this will cause some problem like the algorithm can just set every value to 0. So we modify the equation to $||f(A)-f(P)||^2-||{f(A)-f(N)}||^2+ \alpha<=0$ and prevent the algorithm set every value to 0. The $\alpha$ is also called margin. We can adjust the margin value $\alpha$ to control the contrast.

The function of triplet loss is $L(A,P,N) = max(||f(A)-f(P)||^2-||f(A)-f(N)||^2,0)$

As long as we achieve the goal that $||f(A)-f(P)||^2-||f(A)-f(N)||^2<=0$ , we can just return 0. Since the model don't care how negative it is.

The overall cost function is $J=\sum^m_{i=1}L(A^{(i)},P^{(i)},N^{(i)})$

Training set: 10k pictures for 1k person. Since we do need to have some pair of A and P. We need to have several pictures of a same person.

### Choosing the triplets A,P,N

During training, if A,P,N are chosen randomly,$d(A,P)+\alpha$ is easily satisfied which neural network can not learning anything from it.

Choose triplets that are hard to train on.

We eventually want to have $d(A,P)+\alpha <= d(A,N)$ , we should choose the A and N which $d(A,P)\approx d(A,N)$ , so that the algorithm will try hardly to create this margin $\alpha$ 

It also increase the efficiency of the computation:

- If we choose the A,N randomly, most of them will be meaningless
- If we choose A,N by rules then our neural network will learning within every triplet.



After we have define and create triplets as our training data, we can use gradient descent to minimize the cost function J in order to learning a encoding so that $d(x^{(i)},x^{(j)})$  is small when there are image for same person and vice versa.



## Face verification and binary classification

Face recognition can also be treated as a binary classification

Based on the Siamese network we can also add a logistic unit at the end of the network and treat it as a binary classification problem.  And this is an alternative for triplet loss. 

$\hat y = \sigma(\sum_{k=1}^{128} |f(x^{(i)})_k-f(x^{(j)})_k|)$

This equation take element wise difference and absolute value between these two encoding. We consider these 128 element as feature and then feed into logistic regression. Then we can add weights and biases into previous equation 

$\hat y = \sigma(w_i\sum_{k=1}^{128} |f(x^{(i)})_k-f(x^{(j)})_k|+b)$

We can then train the w and b to make model to learn if these two image about same person.

There also some variation  to compute $|f(x^{(i)})_k-f(x^{(j)})_k|$ 

For example $ \frac{(f(x^{(i)})_k-f(x^{(j)})_k)^2}{f(x^{(i)})_k+f(x^{(j)})_k}$ which is called as $\chi$ similarity 

#### Computational trick

- We can pre-compute the encoding for  anchor image  and just compare to the input image. 
- In this way we don't need to store the image, we can just store the encoding.
- This will save computational power and make it more efficient.
- This trick works good at both Siamese and logistic regression.



## Neural style transfer

![](img\Neural Style transfer.png)

$



## What are deep ConvNets learning

 

- Pick a unit in layer 1. Find the nine image patches that maximize the unit's activation. 
- Go through the entire training set, find the image or image patch that maximize the activation unit.
- Repeat operation to other units in the same layer.

For units in the layer one we are only able to observe very little portion of the neural network.  When we visualize the output we can see that units in layer one are going to learning some simple edges or color shade.

![](img\Convisual0.png)

As the layer went deeper and deeper, units starts to learn more complex features.

![](img\convisual1.png)

## Cost function for image.

### Neural style transfer cost function 

$J(G) = \alpha J_{content}(C,G)+\beta J_{style}(S,G)$

We have $\alpha $ and $\beta$ to specify the relative weighting between the content cost and styling cost.

Two hyperparameter seems to be redundant.

### Find the generated image G

1. Initiate G randomly

   G:100x100x3

2. Use gradient descent to minimize J(G)

$G:= G-\frac{\partial}{\partial G}J(G)$

When we initialize the G randomly, we will have a picture full of noises.

The gradient descent will treat every pixel and then make the image looks like the content image with the style in style image.

## Content cost function

$J(G) = \alpha J_{content}(C,G)+\beta J_{style}(S,G)$

- Say you use hidden layer l to compute content cost.
- This layer l should not be too shallow or too deep. If L is layer 1 then it means the output image will be very similar to content picture at pixels. If we used deep layers, then it might be to make sure that they have the same object.(Edge,dog,cat,person.)
- Use pre-trained Convnet.(E.g., VGG network)
- Let $a^{[l](c)}$ and $a^{[l](c)}$ be the activation of layer l on the images.
- If $a^{[l](c)}$ and $a^{[l](c)}$ are similar, both image have similar content.
- $J_{content}(C,G)=\frac{1}{2}||a^{[l](c)}-a^{[l](c)}||^2$
- Then we run gradient descent on this cost function J and this will incentivize the algorithm to find the image G that has similar content as Content image.



## Style cost function

### Meaning of the "Style" of an image

Define style as correlation between activations across channels.

How correlated are the activations across different channels. In the following picture, we can check if vertical edges tend to show up with orange color, after we find the correlation between these features we can generate the image with same style.

![](img\neural_style_transfer0.png)

![](img\Neural_style_transfer.png)

### Style matrix

Let $a^{[l]}_{i,j,k}$ = activation at (i,j,k). $G^{[l](s)} is\ n^{[l]}_c\times n^{[l]}_c$ which i and j is for height and weight and k for channels. 

$G^{[l](S)}_{kk'}$ is used to describe how correlated our channels k and k' in following equation.

$G^{[l](S)}_{kk'}=\sum_i^{n_w}\sum_j^{n_h}a^{[l]}_{ijk}a^{[l]}_{ijk'}$

k and k' will be ranged from 1 to n_c

This equation is the unnormalized cross correlation because we don't subtract the mean. 

Then we do the same thing to generate image

$G^{[l](G)}_{kk'}=\sum_i^{n_w}\sum_j^{n_h}a^{[l]}_{ijk}a^{[l]}_{ijk'}$

This also called Gram matrix.

Then we can have the style cost function:

 $J^{[l]}_{style}(S,G)= ||G^{[l](s)}-G^{[l](G)}||^2$

$J^{[l]}_{style}(S,G)=\frac{1}{2n^{[l]}_hn_w^{[l]}n_c^{[l]}}\sum_k \sum_{k'}(G^{[l](S)}_{kk'}-G^{[l](G)}_{kk'})$

The normalized term can be omitted since there is hyperparameter to control the weight of it.

If we used every this style cost function to every layer, the performance will be better.

$J_{style}(S,G) = \sum_l\lambda^{[l]}J^{[l]}_{style}(S,G)$

Then we will apply both content cost function and style cost function into the overall cost function and run gradient descent.

$J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)$

Then we can get a image we want.



## Convolutions in 2D and 1D

Convolution can also be applied to 1D and 3D data. Here are some example:

- 2D convolutions: 14x14x3 convoluted with 5x5x3 = 10x10xnc
- 1D convolutions: 14x1 convoluted with 5x1 = 10 xnc



3D volume data（like CT scan）：

Then we will have 14x14x14x1 convoluted with 5x5x5x1  and we have 16 filters

We will have 10x10x10x16







