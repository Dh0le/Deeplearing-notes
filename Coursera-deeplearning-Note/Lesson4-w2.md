# Deep CNN case study

##  LeNet-5

LeNet-5 was trained for greyscale image and that is the reason why has 32x32x1 only 1 channel. 

- Since the padding is not popular at that age, the size of height and weight is decreasing. With network goes deeper and deeper the number of channel increases. 
- Every conv layer is followed by a pooling layer. 
- Average pooling is often used in LeNet-5 istead of maxpooling.
- In LeNet, activation function are mostly: **Sigmoid and Tanh**

![](img\20171114151228414.png)



## AlexNet

- Comparing to LeNet-5, the structure of network is bigger,with more parameters and channels.  So the performance is better.
- AlexNet starts to use ReLU
- Back in that times, The computational power of GPU is still very low so the Alex Net actually use multiple GPUS
- The use of LRN(Local response normalization) was discard lately since the performance is very limited.

![](img\20171114152041555.png)



## VGG-16 Net

- Less hyperparameters
- Conv =3x3 s=1,same
- Max-Pool =2x2 s =2
- Its a relatively deeper network with 1.6 billion parameters.

![](img\VGG.png)



## Residual Networks (ResNets)

ResNets is build by Residual blocks. What is Residual Blocks

#### Residual Block

Lets first take a look at the plain neural network.(The network without residual block)

![](img\20171114153936535.png)

The forward propation of this process is:

- Linear: $z^{[l+1]}=W^{[l+1]}a^[[l]]+b^{[l+1]}$
- Activation: $a^{[l+1]} = g(z^{[l+1]})$
- Linear: $z^{[l+2]} = W^{[l+2]}a^{[l+1]}+b^{[l+2]}$
- Activation:$a^{[l+2]}=g(z^{[l+2]})$



In the resnet, we add a shortcut that link $a^{[l]}$ directly to $z^{[l+2]}$.

The last equantion becomes $a^{[l+2]}=g(z^{[l+2]}+a^{[l]})$

![](img\20171114155305863.png)

Please **Note**: The link is before activation function



#### Link of the residual blocks

![](img\20171114155750281.png)



Residual network is great way to deal with degradation problem

![](img\20171114160023551.png) 

 

# Why do neural networks work?

If we have already a big network and we want to make it deeper. The best way is to add a residual block.

![](img\resualblock.png)

Assume that we use ReLU activation function so that the equation of $a^{[l+2]}$ is :

$a^{[l+2]}=g(z^{l+2}+a^{[l]})=g(W^{[l+2]}a^{[l+1]}+b^{[l+2]}+a^{[l]})$

If we are using L2 norm of weight decay, then the equation will shrink the value of $W$ and $b$. Then the previous equation will become :

$a^{[l+2]}=g(z^{[l+2]}+a^{[l]})=a^{[l]}$

This equation is pretty easy for network to learn and it wont cause degradation problem while increase the depth of network. 

And if the layer you added actually learn some features, the performance of the network will increase.

In order to keep $a^{[l+2]}=g(z^{l+2}+a^{[l]})$, ResNet broadly use **SAME** padding to keep the size.

![](img\plaintoRes.png)



Sometimes we need to use $W_s$ or padding after some pooling layers.



# 1x1 Convolution(Network in Network)

The 1x1 convolution in 2D is just the sum of all element and filter weight.

But in 3D the convolution operation with 1 x 1 x nc is to do slice to the 3D matrix is essentially create fully connect network through every slice and apply with ReLU function.

So the output of  $6\times6 \times32$ matrix convoluted with n $1\times 1\times32$  filters will output $6\times 6\times n$ 

This operation outputs matrix with same height and width but different channels.

![](img\NetworkinNetwork.png)

### 1x1 Conv application

1. Change dimension: 1x1 conv can change the number of channel base on filter number. You can increase or decrease channel number base on demand.
2. Add nonlinearity: It is okay for you to not to change the dimension and still output in same size. This **Network in network ** will simply add nonlinearity to the model.

![](img\NinN2.png)





# Inception network motivation

The motivation of Inception network is to help developer who  don't want to choose hyperparameter like filter size. The key  idea of this Inception network is to stack up the result of different layers. So we can just let machine to choose which filter it could use.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\Coursera-deeplearning-Note\img\Inception2.png)

As we can tell from the previous picture. In order to keep the size of each filter, **SAME** padding are used in this case.

The convenience does not come for free. The computational cost for inception network is pretty unacceptable even for modern computers.

Taking the 5by5 filter as example: 

![](img\Inception2.png)

as we can see, the computations for this filter is 120 million.

The method we can use to lower the number of computations is by introduce **Network in network** which is the 1x1 conv layer we previously mentioned.

By using 1x1 conv layer we can successfully reduce the number of  computations.

![](img\Inception3.png)

As we can see, 1x1 conv layer(bottleneck layer) successfully reduce the channel numbers and therefore reduce the number of computations to 12.4 million.

**Notation:**

It is possible for developer to shrink the number of computation without hurting the performance of the model as long as the implementation is within a reason.



# Inception module

Combining the design we mentioned in previous part. We will successfully create inception module.

![](img\InceptionM.png)

Noticed that max pooling will still have the same number of channels as input. So we will do another conv layer after that to get the proper size we want.

### Inception network

Inception network is basically build by Inception module. 

![](img\InceptionNN.png)

As you can see in the picture, there are multiple inception modules that connected this network. Between each inception modules, there are some max-pooling layers to control the size of input.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\Coursera-deeplearning-Note\img\SideChannel.png)

 

As we can see, there are some side branches that output result through FC layers and softmax.  These side branches make sure that even the intermediate layers are not too bad at predicting class of image. This is a part of regularizing effect in inception network that helps developer to prevent overfitting. 

# Transfer Learning

My researchers will open source their work into opensource communities. It is an efficient way for developers to download pre-trained network and weight and train their own network on it. This is a great solution to deal with lack of data problem. We call this method **Transfer learning**.

1. Typically, we delete the final output layer like softmax and just add our own softmax layer and weight into it. Freeze the previous layers and weight we download from others work and just train the last layer.
2. We can also calculate the input through the network of others and store the output into disk. Then the training process will not have to go through previous network in every epoch. And therefore save the computational power.

3. With data set you have getting larger and larger. You can freeze several early layers and train the rest of the network. Or just simply use the weight of previous network to instead of initialization.
4. When you have very large data, you can just use weight other previous network as the initialization. And then do gradient descent.

# Data augmentation

The most important problem for CV area is that we cannot have enough data to train the model. And that is why data augmentation is working. Because it allows us to create more data form current data.

Popular method of data augmentation:

- Mirroring
- Random Cropping(It is no a perfect way, since it can crop some undesired part of the image.)

- Color shifting. By adding different number to different channels to create picture in different color shade. PCA color augmentation. (Which will change more to the major color of the picture, less to secondary color so that the overall color will stay the same.)

![](img\Distortion.png)

In the data augmentation process, we can use a CPU thread to form mini batch from the augmented data and then pass to the model that is training in GPU or other CPU thread.

# Current CV deep learning

## Data vs hand-engineering

Here is a picture that shows the different data demand with different task.

![](img\DataDemand.png)

When developers have a lot of data, they intend to use less hand-engineering, since developers do not need to carefully design the features or structure. A bigger or deeper network will just solve the problem.

And this situation is reversed if we have a very little data. Hand engineering will out perform ML.

There commonly two sources of knowledge:

- Labeled data
- Hand engineering features/network architecture/other components

Tips for doing well on benchmarks/wining competitions.

- Ensembling:

  - Train several networks independently and average their output. (Average y hat)
  - This means you will need to train more network and go through the entire examples. It might never been used in production.

- Multi-crop at test time

  - Multi-crop is a form of applying data augmentation into test image.
  - Run classifier on multiple version test image and average result. 
  - Computational budget is pretty large. 

  

Use open source code

- Use architecture of networks published in the literature.
- Use open source implementation if possible
- Use pretrained model and fine tune on you datasets

