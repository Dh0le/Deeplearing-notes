# Hyperparameter setting process

## List of hyperparameters

- Learning rate $\alpha$
- Momentum $\beta$
- Adam $\beta_1,\beta_2,\epsilon$
- Number of hidden layer
- Number of hidden unit
- Learning rate decay factor
- Mini-batch size

### Priority rank

1. learning rate $\alpha$
2. Momentum $\beta$,hidden unit, mini-batch size
3. Number of layers, learning rate decay


### Grid vs Random

- Grid used to be a great way to tune very small number of hyperparameters
- Use Random selection will be the new way.
  - Because it is difficult to know in advance which hyperparameter is more important.
- Another common practice is to use a coarse to fine sampling scheme.
  - Zoom in, into a smaller region and do the sampling more densely in that region.

## Using an appropriate scale to pick hyperparameters

- using log scale

  ```python
  r = -4 * np.random.rand() # r will be a random number between 4 and 0
  learning_rate = 10^r # between 10^4 and 10^0
  ```

- Example: we want a to be randomly choose value from 0.0001 and 1
  - We calculate the $a = log_{10}^{0.0001}=-4$ and $b = log_{10}^0=1$
  - Then choose value uniformly between a and b

## Hyperparameters for exponentially weighted average

- $\beta$ value for EWA is 0.9 for basically 10 samples and 0.999 for 1000
- We will sample for $1-\beta$ then the problem will goes for (0.1 -0.001)
- Then just follow the previous steps
- For example $r \in [-3,1]$
- $1-\beta = 10^r$
- $\beta = 1-10^r$
-  The formula for exponentially weighted average is very sensitive to small change when beta is close to 1.
- The sampling algorithm will make the process to sample more densely over the region where beta is close to 1.

# Hyperparameter tuning process in practice

There are mainly two major school of thought

1. Babysitting one model
   - People tends to do this if their have a huge data set but not a lot of computational resources like CPU and GPU that we can only train one model or very small number of models at a time.
   - Babysitting one model, according the curve of cost function J, change each hyperparameter over a day.
2. Train many model in parallel
   - Try different set of hyperparameter at the same time, and compare the  the curve of learning rate and select the best.
3.  Panda vs Caviar



# Batch Normalization

## Normalize input 

- Normalizing inputs to speed up the learning, change something elongated into a more rounded shape for gradient descent
- $\mu = \frac{1}{m}\sum x^i$
- $x = x-\mu$
- $\sigma^2 = \frac{1}{m}\sum x^{(i)2}$
- $x=\frac{x}{\sigma^2}$

## Deeper model- batch normalization

- The key idea of batch normalization is to normalize $a^i$ so we can train $w^{i+1}$and $b^{i+1}$ faster
- there are debates of should we normalize $a^i$ or $z^i$ 
- In practice , normalize the input before activation function is more often.
- So we use $z^i$ in following notes.
- $\mu = \frac{1}{m}\sum z^i$
- $\sigma^2 = \frac{1}{m} \sum(z^i-\mu)^2$
- $z^{i}_{norm} = \frac{z^i-\mu}{\sqrt{\sigma^2+\epsilon}}$
- $\tilde{z}^i=\gamma z^i_{norm}+\beta$
- the $\gamma$ and $\beta$ is the learnable parameters of  your model
- The function of $\gamma$ and $\beta$ is to set the mean of $\tilde{z}$ to be whatever you want it to be.
- The main idea of batch norm is to keep normalization process not only in the input layer but also in other layers.

## Fitting batch norm into a neural network

``` mermaid
graph LR
A[X1]-->|W1,B1|B[Z1]
B-->|beta,gamma,batch norm|C[ztilde]
C-->|activation|D[a1]
D-->|w2,b2|E[Z2]
```

The batch norm happens between z1 and activation

Parameters: $w^1.....w^l$,$b^1....b^l$,$\beta^1...\beta^l$,$\gamma^1.....\gamma^l$

Warning: This $\beta$ and $\gamma$ has nothing to do with momentum.

Then we can use any optimization method to optimize the $\beta,\gamma$

Using framework does not need to implement this all by yourself. It usually would be a single line of code like

``` python
tf.nn.batch_nromalization
```

- Batch norm usually works with mini-batches
- The parameter $b$ (bias) will actually be neutralized by the mean subtraction steps. So if you are using batch norm, bias can actually be cancel.
- The dimension of $\gamma^{[l]},\beta^{[l]}$ will also be {${n^[l],1}$}  

``` pseudocode
for t = 1 .... number of mini batches
	compute forward prop on x^{t}
    	In each hidden layers, use batch norm to replace Z^l with ztilde l
    use back pro to compute the grandient dw,db,dbeta,degamma
    update parameter
```



works with momentum, adam ,rmsprop.



# Why batch norm works?

Learning on shifting input distribution

- It makes weights, later or deeper in the neural network more robust to changes to weight in earlier layers of the neural network.
- Covariate shift 
  - When the distribution of example changed, then you need to retrain the algorithm
  - Even the ground true function is remain the same.
- The batch norm is reduce the amount that the distribution of hidden unit value shifts around. 
- It limited the amount to which updating the parameters in the earlier layers can affect the distribution of values.
- It reduce the problem of the input value changing. It cause these values to be more stable so the later layers would  have more firmed ground to stand on.

### Second effect

It add some noise to the hidden layer activation so it has some regularization effect.

Code example :

``` python
def Batchnorm_simple_for_train(x, gamma, beta, bn_param):
"""
param:x    : input data, shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
	eps      : 接近0的数，防止分母出现0
	momentum : 动量参数，一般为0.9， 0.99， 0.999
	running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
	running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
	running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]
	results = 0. # 建立一个新的变量
    
	x_mean=x.mean(axis=0)  # 计算x的均值
    x_var=x.var(axis=0)    # 计算方差
    x_normalized=(x-x_mean)/np.sqrt(x_var+eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移
	## the running mean will be useful during the test process.
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
    
    #记录新的值
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var 
    
	return results , bn_param
```





## Testing for batch norm

Since the mean and variance are generated in a mini batch, it will be meaningless in the testing process since we only input 1 example.

Solution:

using a estimate a exponential weighted average(across all the mini batches) to keep track the mean and variance.

Add following code into training process 

``` python
running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
```



And the test code should be like

``` python
def Batchnorm_simple_for_test(x, gamma, beta, bn_param):
"""
param:x    : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
	eps      : 接近0的数，防止分母出现0
	momentum : 动量参数，一般为0.9， 0.99， 0.999
	running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
	running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
	running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]
	results = 0. # 建立一个新的变量
   
    x_normalized=(x-running_mean )/np.sqrt(running_var +eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移
    
	return results , bn_param
```



# Softmax regression

Softmax layer will be able to output possibility of multiple categories

the implementation of softmax regression

1. $z^{[L]}= W^{[L]}a^{[L-1]}+b^{[L]}$
2. $t=e^{z^{[l]}}$
3. $a^{[l]}= \frac{e^{z^{[l]}}}{\sum t^i}$
4. $a^{[l]}_i=\frac{t_i}{\sum t_i}$

$a^{[l]}$ is a (c,1) vector

## understand the softmax activation

if the c = 2 then the softmax is equal to logistic regression

The loss function of the softmax activation:

$L(\hat y,y)=-\sum y_jlog(\hat y)j$

Then the total cost function will be 

$J(w^l,b^l...)=\frac{1}{m}L(\hat y,y)$

Y is a (c,m) matrix

Y hat is also a (c,m) matrix

Important:

Back prop for softmax:

$dz = \hat y-y$



## Deep learning framework

Since the course have already gone public for a while, many framework might actually no longer as popular as it was.

Pytorch and Tensorflow might be a great choice for implementing a DNN.

​	



