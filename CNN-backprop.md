# Back propagation for CNN

## Back propagation for convolutional layer

Modern deep learning framework usually takes care of the backprop process. So developers only need to implement forward propagation.  Comparing to the FC， back propagation in convolutional network is bit more complicated.  But the main idea is the same as normal DNN, we can compute derivative towards cost and then use them to update the parameter. 



So as always, we will compute $dA^{[l-1]}$ ,then we need to compute $dW^{[l]}$ and $db^{[l]}$

1. $dA$:

   the way $dA^{[l]}$ is computed is :

   $dA\ += \sum^{n_H}_{h=0}\sum_{w=0}^{n_W} W_c \times dZ_{hw}$

   Which $W_c$ is a filter and $dZ_{hw}$ is a scalar corresponding to the gradient of the cost with respect of the output of conv layer Z at the h row w column. Note that at each time, we multiply the the same filter  $W_c$ by a different $dZ$ when updating $dA$. We do so mainly because when computing the forward propagation, each filter is dotted and summed by a different a_slice. Therefore when computing the backprop for $dA$, we are just adding the gradients of all the a_slices. 

   Following code provides you an helpful example of doing this:

   ```python
   da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
   ```

   

2. $dW$:

   $dW_c$ is the derivative of one filter with respect to the loss. The way we compute $dW_c$ is:

   $dW_c \ += \sum^{n_H}_{h=0}\sum_{w=0}^{n_W} a_{slice}\times dZ_{hw}$

   Where the $a_{slice}$ corresponds to the slice which we have previously used to generate the $Z_{ij}$. Hence, this ends up giving us the gradient for $W$ with the respect to that slice. Since it is the same $W$, we just add up all such gradient to get $dW$.

   Following code provides you an helpful example of doing this:

   ``` python
   dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
   ```

3. $db$:

   This is the formula for computing $db$ with respect to the cost of a certain filter $W_c$:

   $db = \sum_h\sum_wdZ_{hw}$

   As you have previously seen in DNN, compute $db$ is just summing up $dZ$. In this case, we summing over all the gradient of con output Z

   ```python
   db[:,:,:,c] += dZ[i, h, w, c]
   ```



## Back propagation for pooling layer

In the pooling layer, we usually use two type of pooling method

1. Max pooling 

   Max pooling is a very simple method. And it has no parameter for gradient descent to learn. We will implement a helper function to compute what we want:

   $X = \begin{bmatrix} 1 && 3 \\ 4 && 2 \end{bmatrix} \quad \rightarrow  \quad M =\begin{bmatrix} 0 && 0 \\ 1 && 0 \end{bmatrix}$

   To implement this we can use np.max function in numpy lib.

   ```python
   mask = x == np.max(x)
   ## this line of code means
   ## mask[i,j] = 1 if x is the max in x
   ## mask[i,j] = 0 if x is not the max in x
   ```

   

2. Average pooling

   It is slightly different from the max pooling since the the affect toward cost function is averaged over all element in pooling layers. So we will also do that in backprop. 

   $dZ = 1 \quad \rightarrow  \quad dZ =\begin{bmatrix} 1/4 && 1/4 \\ 1/4 && 1/4 \end{bmatrix}$

   ```python
   (n_H, n_W) = shape
       
       # Compute the value to distribute on the matrix (≈1 line)
   average = dz / (n_H * n_W)
       
       # Create a matrix where every entry is the "average" value (≈1 line)
   a = np.ones(shape) * average
   ```

   



After we have these two helper function, we can follow the previous routine to compute the derivative for pooling layers.