# Week2--Optimization

## Mini-batch gradient descent

- ***Batch vs mini-batch gradient descent***
  - Vectorization allows you to efficiently compute on m examples. But when the total number of example is too large it can still be very slow for gradient descent process.
  - Mini-batch gradient descent is a method that helps you to make progress before actually go through the entire training set. Which means it will be faster.
- Split the entire training set and example labels into t mini batches which denoted as $x^ \left\{ t \right\} $ and corresponding  $y^ \left\{ t\right\} $ . Remember to check the dimension of these mini batches

### The process of Mini batch gradient descent

1. Set up a for loop to iterate all the mini batches
2. Do the forward propagation over 1 mini batch
3. Compute the cost function over 1 mini batch
4. Do the back propagation over 1 mini batch to compute the gradient 
5. Perform gradient descent

Mini batch gradient descent will do much more gradient descent steps in 1 epoch. And I will run much more faster than normal gradient descent.

### Choosing mini batch size

- If mini batch size = m : Then it is batch gradient descent   **Low noise and large steps, too long for 1 iteration**
- If mini batch size = 1 : Then it is stochastic gradient descent **Wont ever converge but oscillating in an area, loss all the speed up from you vectorization **
- Mini batch should be in between, not too large/small **Make fastest learning, Vectorization**

### How to choose

- Small training set(m<=2000) : Use batch gradient descent
- Typical mini batch size: power of 2 like 64,128,256,512, 1024.
- Make sure that your batch fit in CPU and GPU memory.
- This is a important hyper parameter
- Try different values.



# Exponentially weighted averages

- Formula : $V_t = \beta v_{t-1}+(1-\beta)\theta_t$ $\theta_t$ is t he actual value and $\beta$ is the weight decrease speed and $V_t$ is the Exponentially weighted moving averages
- $V_t$ as approximately average over $\frac{1} {1-\beta}$ days temperature 
- $\beta$ are more sensitive towards drastic change when small, more smooth and less noisy when large
- During the implementation, we usually keep the $V_t$ update with current $\theta$ value, so it only use very short memory, it is not the best way.
- In mathematic definition, using $\frac {1}{e}$ as threshold 

## Bias correction

- problem happened when we initialize weight to 0, it will cause all the early phase data to be too small
- Change the formula : $V_t = \frac {\beta v_{t-1}+(1-\beta)\theta_t}{1-\beta^t}$
- When t is small the denominator will be big enough to do bias correction. When t $t$ gets large. $\beta^t$ will be extremely small to make any difference. 

# Gradient descent with momentum

- On iteration t:

  - Compute the $dw$ and $db$ on current mini batch
  - $v_{dw} = \beta v_{dw} +(1-\beta)dw$
  - $v_{db} = \beta v_{db} +(1-\beta)db$
  - $w:= w-\alpha v_{db}, b:=b-\alpha v_{db}$
  - $v_{dw}$ and $v_{db}$ should be initialized to 0
  - Hyperparameters : $\alpha,\beta$  0.9 should be a robust value for $\beta$ 

  

# RMSProp-root mean square prop

- On iteration t:
  1.  Compute the $dw$ and $db$ on current mini batch
  2. $ S_{dw} = = \beta_2 v_{dw} +(1-\beta_2)dw^2$
  3. $S_{db} = \beta_2 v_{db} +(1-\beta_2)db^2$
  4. $W:= W-\alpha \frac{dw}{\sqrt{S_{dw}+\epsilon}},b:=b-\alpha \frac{db}{\sqrt{S_{db}+\epsilon}}$ and $\epsilon = 10^{-8}$
- This will have relatively large $S_{db}$ so it can successfully reduce the unnecessary oscillation 
- Can use larger learning rate

# Adam optimization algorithms

1. Initialize $V_{dw}=0$,$S_{dw}=0$,$V_{db}=0$,$S_{db}=0$
2. On iteration t:
   1. Compute the $dw$ and $db$ on current mini batch
   2. $ S_{dw} = = \beta_2 v_{dw} +(1-\beta_2)dw^2$
   3. $S_{db} = \beta_2 v_{db} +(1-\beta_2)db^2$
   4. $v_{db} = \beta_1 v_{db} +(1-\beta_1)db$
   5. $v_{dw} = \beta_1 v_{dw} +(1-\beta_1)dw$
   6. $V^{corrected}_{dw} = \frac{v_{dw}}{(1-\beta^t_1)}$ and $V^{correct}_{db} = \frac{v_{dw}}{1-\beta^t_1}$
   7. $S^{correct}_{dw} = \frac{S_{dw}}{1-\beta^t_2}$, $S^{correct}_{db}=\frac{S_{db}}{1-\beta^t_2}$
   8. $W:=W-\alpha \frac{V^{correct}_{dw}}{\sqrt{S^{corrected}_{dw}}+\epsilon}$
   9. $b:=b-\alpha\frac{V^{corrected}_{db}}{\sqrt{s^{corrected}_{db}}+\epsilon}$

- Hyperparameters choice

  - $\alpha$ needs to be tuned

  - $\beta_1$  is always set to 0.9

  - $\beta_2$ according to the paper of Adam inventor should be 0.999

  - $\epsilon$ , does not really matter $10^{-8}$ should be good

    

# Learning rate decay

- Decrease the learning rate during the later phase of the learning 
- How to implement 
  1. 1 epoch = 1 pass through data se
  2. $\alpha = \frac{1}{1+decayrate*epochnum} * \alpha_0$
  3. Tuning for both learning rate and decay rate

## Other learning rate decay method

- $\alpha=0.95^{epochnum}*\alpha_0$ - exponentially decay
- $\alpha = \frac{k}{\sqrt{epochnum}}*\alpha_0$
- Discrete staircase

# Local optima 

- In high dimensional graph, the 0 gradient usually to be the saddle point
- Low dimensional space experience does not transfer to high dimensional space
- Problem of plateaus
  - Unlikely to get stuck  in a bad local optima
  - Plateaus can make learning slow



