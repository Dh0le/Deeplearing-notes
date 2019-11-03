# Activation function look book

This article is mainly based  on an article of Casper Hansen at 22 Aug 2019 and several other materials. Chinese version will be released later.



The prerequisite for this article is about forward propagation and backward propagation. Please make sure you understand how these processes work and the role played by activation function in these processes.



Before we actually go to talk about activation functions. We will go through some common problems that related to activation function. 

## Vanishing Gradients Problem

If you are familiar with back propagation and updating the weights, you will understand the following equation

$w^{(L)}=w^{(L)}-learning\ rate \times \frac{\partial C}{\partial w^{(L)}}$

and the $\frac{\partial C}{\partial w^{(L)}}$ was calculated by the back propagation process, and that is the key factor of how can we minimize the cost function.

But what if the $\frac{\partial C}{\partial w^{(L)}}$ is extremely small? That is what we called vanishing gradient problem, and this problem will cause the weight and bias in corresponding layers to have very little updates. And the vanishing gradient problem will holds back the model form learning.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\Activation-functions--2-.png)

And vanishing gradient problem has more effect towards the model that trains different layers with different learning speed.

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\image-2.png)

I hope following equations of gradients in different layers will give you an answer of why this happened.

$\frac{\partial C}{\partial w^{(1)}}    =    \underbrace{    \frac{\partial C}{\partial a^{(4)}}    \frac{\partial a^{(4)}}{\partial z^{(4)}}    }_\text{From $w^{(4)}$}    \,    \underbrace{    \frac{\partial z^{(4)}}{\partial a^{(3)}}    \frac{\partial a^{(3)}}{\partial z^{(3)}}    }_\text{From $w^{(3)}$}    \,    \underbrace{    \frac{\partial z^{(3)}}{\partial a^{(2)}}    \frac{\partial a^{(2)}}{\partial z^{(2)}}    }_\text{From $w^{(2)}$}    \,    \frac{\partial z^{(2)}}{\partial a^{(1)}}    \frac{\partial a^{(1)}}{\partial z^{(1)}}    \frac{\partial z^{(1)}}{\partial w^{(1)}}$

$\frac{\partial C}{\partial w^{4}}    =    \frac{\partial C}{\partial a^{4}}    \frac{\partial a^{4}}{\partial z^{4}}    \frac{\partial z^{4}}{\partial w^{4}}$



## Exploding Gradient Problem

Exploding gradient problem is actually the opposite of vanishing problem. The problem is shown and done by Nielsen. We are most likely to run into a vanishing gradient problem when we have $0<w<1$ and exploding gradient problem when we have $w>1$ 

If you are interested in the mathematic intuition of how the exploding gradient problem happened. Please visit  ["Why are deep neural network hard to train"](http://neuralnetworksanddeeplearning.com/chap5.html) here.

But don't worried, we do have some methods that help developers to avoid exploding gradient problem.

## Gradient clipping/norms

The idea behind gradient clipping/norms is to setting up a rule that helps the model to correct the gradient while it might goes into an exploding gradient problem. We will skip the mathematical detail and here is two basic steps of implementing gradient clipping/norms.

1. Pick a threshold value - if a gradient passes this threshold value, we apply gradient clipping/norms.
2. define if you want to use gradient clipping or norms. If you want to choose gradient clipping, you will need to specify a threshold value, if the gradient is exceeding that value then it will be set back to that threshold value. And if you want to use gradient norm, than  you will scale back the gradient.

You will need these methods while developing a Recurrent Neural Network like LSTM or GRU, since there is high probability of experiencing exploding gradient problem.



After we discuss about the methods that we can use to solve exploding gradient problems, we focus back on vanishing gradient problem and the first and most common activation function **ReLU**.



## ReLU



Rectified Linear Unit. This is an activation function that helps us with gradient vanishing problem.

The equation of ReLU is :$ReLU(x)=max(0,x)$

And plot of the ReLU function is 

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\relu.png)



According to the definition of ReLU, we can tell it is very simple. But how can this simple equation help us with vanishing gradient problem?

To know that, we will need to take a look at the differentiated equation:

$\text{ReLU}'(x) =    \begin{cases}    \mbox{$1$} & \mbox{if } x > 0\\    \mbox{$0$} & \mbox{if } x \leq 0    \end{cases}$

Which means that we will also get 1 if x is greater than 0 and 0 if x is less or equal to zero.

Here is the graph of ReLU differentiated:

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\relud.png)

And that is the reason why  we can avoid vanishing gradient problem. Because it will only gives you 0 or 1, it will never generate a extremely small value like 0.0000000023.

### Dead ReLU problem

This problem happens when there are too many values below 0 and many weight and bias will not be updated. But there are actually a good aspect of this problem which is called Sparsity.

#### Sparsity

- Small in numbers or amount, often spread over a large area. In the neural network this means that the matrices for the activations have many 0s. When a certain percentage  of activation are saturated, we call this neural network sparse. 

Why this is a good respect? 

Sparsity leads to an increase in efficiency with regards to time and space complexity, since constant values often requires less space and computational power. This has been observed by Yoshua Bengio.



### Summary

Pros: 

-  Less time and space complexity, because of sparsity and compare to other non-linear activations, it dose not involve the exponential operation.
- Avoids the vanishing gradient problem.

Cons:

- Dead ReLU problem, some weight and bias might never get updated.
- This cannot avoid the exploding gradient problem.



## ELU

Exponential Linear Unit.  This activation function fixes some problem with ReLU and keeps some of its advantages.  

- You have to pick an $\alpha $ value which is commonly between 0.1 and 0.3

Lets take a look at its equation and graph

$\text{ELU}(x) =    \begin{cases}    \mbox{$x$} & \mbox{if } x > 0\\    \mbox{$\alpha (e^x-1)$} & \mbox{if } x < 0    \end{cases}$

For ELU, it is the same while $x$ is greater than 0 and we go slightly below 0 while $x$ is less than 0.

The output while x is less than 0 is depending on both $x$ value and $\alpha$ value. Since ELU involves $e^x$ this is more computationally expensive than the ReLU function.

Here is the graph of ELU while $\alpha=0.2$

 ![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\elu.png)

This activation is just slightly different and more complex than the ReLU. It should not be hard to understand. And lets move to the differentiated part.

Here is the differentiation function of ELU:

$\text{ELU}'(x) =    \begin{cases}    \mbox{$1$} & \mbox{if } x > 0\\    \mbox{ELU$(x) + \alpha$} & \mbox{if } x \leq 0    \end{cases}$



and the graph of it is:

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\elud.png)



By adding exponential function, ELU successfully avoid **dead ReLU** problem. However, since the output of it is still constant, and we still keep some dead components in the network, the computational speed gain by the ReLU was maintained.



### Summary

Pros:

- Produces negative outputs and helps the network nudge weight s and bias in the right direction.
- Avoid the dead ReLU problem
- Produces activations instead of letting them to be zero when calculating the gradient.

Con:

- Requires more computational power.
- Does not avoid exploding gradient problem
- The neural network dose not learn the alpha value
- This was not mentioned in the original article, but we do have one more hyperparameter $\alpha$. Since the ideally range was given, I believe it wont be a bigger problem.



## Leaky ReLU

Leaky Rectified Linear Unit is some how like the ELU, it also requires an $\alpha$ value. Commonly $\alpha$ will be in between 0.1 and 0.3

Lets take a look at the equation and graph of this Leaky ReLU

$\text{LReLU}(x) =    \begin{cases}    \mbox{$x$} & \mbox{if } x > 0\\    \mbox{$\alpha x$} & \mbox{if } x \leq 0    \end{cases}$

Lets assume that we have $\alpha =0.2$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\leaky-relu.png)



I believe the graph and equation are really straight forward. 

Lets take a look at its derivative equation and graph:

$\text{LReLU}'(x) =    \begin{cases}    \mbox{$1$} & \mbox{if } x > 0\\    \mbox{$\alpha$} & \mbox{if } x \leq 0    \end{cases}$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\LRELU.png)

### Summary

Pros:

- Just like ELU, Leaky ReLU can be used to avoid dead ReLU problem.
- It is much faster than ELU since it does not involve with exponential function

Cons:

-  Does not avoid the exploding gradient problem 
-  The neural network does not learn the alpha value 
-  Becomes a linear function, when it is differentiated, whereas ELU is partly linear and nonlinear. (This is the tradeoff of second term in Pros)



## SELU

Scaled Exponential Linear Unit.  SELU is one of the most new activations. The original paper that introduce SELU is about 200 pages with tons of theorems and proofs. I will put the link [here](https://arxiv.org/pdf/1706.02515.pdf) in case somebody wants to read it.  When using this activation in practice you have to use ***lecun_norma*l** for weight initialization. If you want to apply dropout you have to used ***AlphaDropout***   

The author of the paper calculate two values $\alpha$ and $\lambda$ for the equation

$a \approx 1.6732632423543772848170429916717$

$\lambda \approx 1.0507009873554804934193349852946$

I am not going to explain or prove how these two values come from. If you are really interesting in those value, I highly recommend you to read the original paper at the link above.

Since these two values are predefined, you don't need to worried about it. Just use them.

Lets take a look at it equation and graph:

$\text{SELU}(x) = \lambda    \begin{cases}    \mbox{$x$} & \mbox{if } x > 0\\    \mbox{$\alpha e^x-\alpha$} & \mbox{if } x \leq 0    \end{cases}$

Please note that there is a $\lambda$ before the bracket. That is, if the input value x is greater than zero, the output value becomes x multiplied by lambda λ. If the input value x is less than or equal to zero, we have a function that goes up to 0, which is our output y, when x is zero. Essentially, when x is less than zero, we multiply alpha with the exponential of the x-value minus the alpha value, and then we multiply by the lambda value. 

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\selu.png)



#### The special case for SELU

I will just quote the explanation from Casper Hansen

> The SELU activation is self-normalizing the neural network; and what does that mean?
>
> Well, let's start with, what is normalization? Simply put, we first subtract the mean, then divide by the standard deviation. So the components of the network (weights, biases and activations) will have a mean of zero and a standard deviation of one after normalization. This will be the output value of the SELU activation function.
>
> What do we achieve by mean of zero and standard deviation of one? Under the assumption, that the initialization function *lecun_normal* initializes the parameters of the network as a normal distribution (or Gaussian), then the case for SELU is that the network will be normalized entirely, within the bounds described in the paper. Essentially, when multiplying or adding components of such a network, the network is still considered to be a Gaussian. This is what we call normalization. In turn, this means that the whole network and its output in the last layer is also normalized.
>
> How a [normal distribution looks](https://www.geogebra.org/m/QEayZCpM) with a mean μμ of zero and a standard deviation σ of one.
>
> ![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\normal-dis--1-.png)
>
> The output of a SELU is normalized, which could be called *internal normalization*, hence the fact that all the outputs are with a mean of zero and standard deviation of one, as just explained. This is different from *external normalization*, where batch normalization and other methods are used.
>
> Okay, great, the components are normalized. But how does it actually happen?
>
> The simple explanation is that variance decreases when the input is less than zero, and variance increases when the input is greater than zero – and the standard deviation is the square root of variance, so that is how we get to a standard deviation of one.
>
> We get to a mean of zero by the gradients, where we need some positive and some negative, to be able to shift the mean to zero. As you might recall ([per my last post](https://mlfromscratch.com/neural-networks-explained/)), the gradients adjust the weights and biases in a neural network, so we need some negative and positive from the output of those gradients to be able to control the mean.
>
> The main point of the mean *mu* μμ and variance *nu* νν, is that we have some domain omega Ω, where we always map the mean and variance within predefined intervals. These intervals are defined as follows:
>
> $\text{mean} = \mu \in \left[ -0.1,\, 0.1 \right]$
> $\text{variance} = \nu \in \left[ 0.8,\, 1.5 \right]$
>
> The ∈ symbol just means that the mean or variance is within those predefined intervals. In turn, this causes the network to avoid vanishing and exploding gradients.
>
> I want to quote from [the paper](https://arxiv.org/pdf/1706.02515.pdf), one which I find important and how they got to this activation function: 
>
> >  SELUs allow to construct a mapping g with properties that lead to SNNs [self-normalizing neural networks]. SNNs cannot be derived with (scaled) rectified linear units (ReLUs), sigmoid units, tanh units, and leaky ReLUs. The activation function is required to have (1) negative and positive values for controlling the mean, (2) saturation regions (derivatives approaching zero) to dampen the variance if it is too large in the lower layer, (3) a slope larger than one to increase the variance if it is too small in the lower layer, (4) a continuous curve. The latter ensures a fixed point, where variance damping is equalized by variance increasing. We met these properties of the activation function by multiplying the exponential linear unit (ELU) [7] with λ>1λ>1 to ensure a slope larger than one for positive net inputs. 

And here is how the differentiated function and graph look like:

$\text{SELU}'(x) = \lambda    \begin{cases}    \mbox{$1$} & \mbox{if } x > 0\\    \mbox{$\alpha e^x$} & \mbox{if } x \leq 0    \end{cases}$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\relu--6-.png)

### Summary

Pros:

-  Internal normalization is faster than external normalization, which means the network converges faster. 
-  Vanishing and exploding gradient problem is *impossible*, shown in the original paper.

Cons:

-   Relatively new activation function – needs more papers on architectures such as CNNs and RNNs, where it is comparatively explored. 



## GELU

Gaussian Error Linear Unit. An activation function used in the most recent Transformers – Google's BERT and OpenAI's GPT-2. The paper is from 2016, but is only catching attention up until recently. 

Lets take a look at its equation:

$\text{GELU}(x) = 0.5x\left(1+\text{tanh}\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right)$

We can see the equation of GELU is just the combination of $tanh$ and a approximated numbers. 

And here is the graph of it:

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\gelu.png)

 It has a negative coefficient, which shifts to a positive coefficient. So when x is greater than zero, the output will be x, except from when x=0 to x=1, where it slightly leans to a smaller y-value. 

And lets check out the differentiation function and graph:

$\text{GELU}'(x) = 0.5\text{tanh}(0.0356774x^3 + 0.797885 x) + (0.0535161 x^3 + 0.398942 x) \text{sech}^2(0.0356774x^3+0.797885x)+0.5$

(Thanks to WolframAlpha)

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\gelu-diff.png)



### Summary

Pros:

- GELU is the state of the art in NLP, especially in transformer model.
- Avoid vanishing gradient problem

Con:

- Newly introduced in practice.



## Sigmoid



Sigmoid function is a logistic function, which means it will only output value between 0 and 1, no matter what you input is.  Which means your input will be scaled to a value between 0 and 1.

Lets take a look at its equation and graph:

$\text{sigmoid}(x) = \sigma = \frac{1}{1+e^{-x}}$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\sigmoid--1-.png)



This function used to be the first nonlinear activation that every new comer learned. T

And lets take a look at its differentiation equation and graph:

$Sigmoid'(z) = Sigmoid(z) \cdot (1 - Sigmoid(z))$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\sigmoid_prime.png)

### Summary

Pros:

-  It is nonlinear in nature. Combinations of this function are also nonlinear.
-  It will give an analog activation unlike step function. 
-  It has a smooth gradient 
-  It’s good for a classifier. 
-  The output of the activation function is always going to be in range (0,1) compared to (-inf, inf) of linear function. So we have our activations bound in a range. Nice, it won’t blow up the activations then. 

Cons:

-  Towards either end of the sigmoid function, the Y values tend to respond very less to changes in X. 
-  It gives rise to a problem of “vanishing gradients”. 
-  Its output isn’t zero centered. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder. 
-  Sigmoids saturate and kill gradients. 
-  The network refuses to learn further or is drastically slow ( depending on use case and until gradient /computation gets hit by floating point value limits ). 



## Tanh 



Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear. But unlike Sigmoid, its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity. Nowadays developer mostly use this Tanh activation in the hidden layers to replace the sigmoid activation.

Lets take a look at its equation and graph:

$tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\tanh.png)



I believe the equation and the graph are pretty straight forward comparing to sigmoid functions. 

Lets move on to the differentiation equation and graph:

$tanh'(z) = 1 - tanh(z)^{2}$

![](E:\Edu\MarkdownNotes\Deeplearning-notes\img\tanh_prime.png)



Based on the graph we can tell that the tanh function does perform better than sigmoid function in some case during the gradient descent.



### Summary

Pros:

- The gradient is stronger for tanh than sigmoid since its derivative is steeper.

Cons:

- We do have vanishing gradient problem while using Tanh



## Softmax

Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will be helpful for determining the target class for the given inputs. 

Lets take a look at its equation:

$ \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $

whereas i  is the number of example and J is the number of classes.

