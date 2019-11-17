## Train/dev/test set

Normally, we do some partition toward our data set in order to train and test a model.

- Training set: The data set that we used to train our model.
- Development set: Which is also called cross validation-set, the key idea of having this model is to get the help developer to select the best model.
- Test set: For final testing, no bias evaluation.

In traditional data era: which we usually get data like 100, 1000,or 10000. The way we split data is:

70%/30% for no dev set.

60%/20%/20% for having dev set.

The situation changed during the era of big data, while we can have like millions of data. We no longer needs to set that big proportion of data for dev and test set.

So we actually come up with some new partition rule:

For millions level data:

98%/1%/1%  for training/dev/test

And for over millions level data:

99.5%/0.25%/0.25% or  99.5% / 0.4% / 0.1%  for training/dev/test

### Notation:

- Please make sure that the distribution of each data set are the same.

- It is okay to not having a dev set if you don't need to compare the performance of different models.



## Bias vs Variance

Here is the picture about the bias and variance problem.

![](img\20170928161736059.png)

According to this picture, we can tell that  **High bias problem** often happens with underfitting. **High variance problem** often happens with over fitting.

There is trade off between bias and variance.

Possible solution for bias:

- Increase layer of neural network
- Increase hidden units.
- Maybe use bigger network.



Possible solution for variance:

- Getting more data.
- Perform regularization
- Select more compact structure.





## Regularization

Regularization is a useful method to deal with high variance problem.

There are two types of regularization:

$l1-norm$ and $l2-norm$

For $l1-norm$:

The model of L1 norm is called Lasso regression. 

$\frac{\lambda}{2m}||w||_1=\frac{\lambda}{2m}\sum^{n_x}_{j=1}|w_j|$

which is the sum of all the absolute value of weight.

For $l2-norm$:

The model of L2 norm is called Ridge regression.

$\frac{\lambda}{2m}||w||_2^2=\frac{\lambda}{2m}\sum^{n_x}_{j=1}w_j^2=\frac{\lambda}{2m}w^Tw$

which is to sum the square of all the weight and do square root.

$J(w..b)=\frac{1}{m}\sum^{m}_{i=1}l(\hat y^{(i)},y^{(i)})+\frac{\lambda}{2m}\sum^{L}_{l=1}||w^{[l]}||$

and also called Frobenius norm.

### The intuition of regularization

If the regularization factor $\lambda$ is big enough, in order to minimize the cost function, the weight matrix $W$ will be set to as small as possible. This leads to a situation that effects of many neuron were shrink thus prevent overfitting problem. 



## Dropout

The key idea of dropout is that we set a threshold value. And according to that value we randomly zero out the effect of certain neurons. Then we will have a relatively smaller neural network.

#### Implementation of dropout

````python
keep_prob = 0.8	
d3 = np.random.rand(a3.shape[0],a3.shape[1])<keep_prob
a3 = np.multiply(a3,d3)
a3 /= keep_prob
````

We explain here why we having a3 /=keep_prob:

According to the implementation above, keep_prob = 0.8 which means 20% neurons were deleted. But we do have calculation $Z^{[4]}=W^{[4]}a^{[3]}+b^{[4]}$, in order not to affect the expectation value of $Z^{[4]}$ so we need to divide keep_prob value.

**Notation** : Please don't used drop out during the testing stage since that will make the result random.

#### The intuition of Dropout

After adding dropout into the neural network, every input feature do have possibility to be dropped. So the neuron will not be over dependent on certain input feature. By going through the process of propagation drop out will generate the similar effect of $l2-norm$.  

The setting of keep prob factor also varied with number of neurons in certain layers. If  the layer has less neurons, we tend to set keep prob to 1, and for those layer with more neurons, we will set keep prob factor to a relative small value.

Cons：

We do have a drawback with dropout:

Cost function are no longer well-defined. Since every iteration will randomly zero out some neurons, we can no longer plot the learning curve according to iteration.



**Notation** : 

1. Set keep prob to 1(close drop out)
2. Run the model to make sure that the learning curve is converging.
3. Open dropout.



## Other method of regularization 

1. Data augmentation :

   The key idea of data augmentation is to get more data to prevent overfitting.

2. Early stopping:

   The key idea of early stopping is to stop the learning process before the error on dev set start to increase. But early stopping cannot solve the problem that we can not find optima bias variance ratio.

## Normalize input

The steps to normalize input:

1. Calculate the mean value of every example: $\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$
2. Subtract the mean to get the symmetric distribution : $x:=x-\mu$
3. Normalize variance : $\sigma^2=\frac{1}{m}\sum^m_{i=1}x^{(i)^2},x=\frac{x}{\sigma^2}$

#### Why normalize input?

![](img\20170928224929150.png)



According to the image above,  we can tell that the shape of cost function change a lot after we normalized input. After we normalized the input, we can use relatively fewer times of iteration to get to the optima.

## Exploding gradient and vanishing gradient problem

Please go to the [Activation part]( Activation function look book.md ) to see the definition of these two problems.

In order to alleviate exploding/exploding gradient problem, we can use Xavier initialization.

$Var(w_i) = \frac{1}{n}$

````python
WL = np.random.randn(WL.shape[0],WL.shape[1])* np.sqrt(1/n)
````

The reason of doing this is because if the input $X$ of activation function set to have 0 means and 1 variance, output $z$ will also be scaled to same the same range. This method cannot totally solve the exploding/vanishing gradient problem but it dose alleviate them.

Different Xavier initialization for different activation function:

- $Var(w_i)=\frac{2}{n}$ for ReLU
- $Var(w_i)=\frac{1}{n}$ for tanh
- $n$ is the number of neurons.

## Gradient checking

![](img\20170929105544463.png)

According to this page, we can tell that the two sided approx is more accurate than one sided approx. 

Connect all the weight and bias:

Take $W^{[1]},b^{[1]}...W^{[l]},b^{[l}$ and reshape into a big vector $\theta$

Take $dW^{[1]},db^{[1]}...dW^{[l]},db^{[l}$ and reshape into a big vector $d\theta$



![](img\20170929112444083.png)

Check if $d\theta_{approx}\approx d\theta$ 

$\frac{||d\theta_{approx}-d\theta||_2}{||d\theta_{approx}||_2+||d\theta||_2}$



**Notation** :

- Only use gradient checking in the debugging process.
- If gradient checking turns to be wrong, go check every item and see which one has bigger difference.
- Please remember regularization term
- Gradient checking cannot run with dropout
- Run at random initialization; perhaps again after some training.(This situation is rare and it could happen when you gradient descent is working while $W$ and $b$ is close to zero and fail to do what it mean to do when $W$ and $b$ go large.)