# Low level Tensorflow API for TF2

## Tensor and Operation

### Tensor

````python
t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
````

This code will generate a tensor which shape=(2,3),dtype=float32.

### Indexing

````python
t[:,1:]
````

This code will return 2 to final columns. In this case is 

```python
([[2., 3.],[5., 6.]]
```



````python
t[..., 1, tf.newaxis]
````

This code will return the second column of every row and make it a (n,1) vector

In this case it is:

```python
<tf.Tensor: id=313, shape=(2, 1), dtype=float32, numpy=
array([[2.],
       [5.]], dtype=float32)>
```

The following code will help you to understand tf.newaxis better:

````python
feature = np.array([[1,2,3],
                   	[2,4,6]])
center = np.array([[1,1,1],
				   [0,0,0]])
tf_feature = tf.constant(feature)[:,tf.newaxis]
tf_center = tf.constant(center)[tf.newaxis,:]
#The expect shape of tf_feature is  (2,1,3)
#The expect shape of tf_center is (1,2,3)
print(tf_feature)
"""
Expected output is :
tf.Tensor(
[[[1 2 3]]

 [[2 4 6]]], shape=(2, 1, 3), dtype=int32)
"""
print(tf_center)
"""
Expected output is :
tf.Tensor(
[[[1 1 1]
  [0 0 0]]], shape=(1, 2, 3), dtype=int32)
"""
````

### Operations

This is pretty simple

```python
tf.square(t) ###Perform elementwise squre
t+10 ###Perform elementwise adding
t @tf.transpose(t) ### t*t.transpose
```



### To/From NumPy

```Python
t.numpy() ### Return a numpy array
```

```python
a = np.array([[1., 2., 3.], [4., 5., 6.]])
tf.constant(a) ### Generate a tensor according to input numpy array
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
np.square(t) ### generate a squared numpy array according to input tensor
```



### Scalars

```python
t = tf.constant(2.718)
t ##Expected output
#<tf.Tensor: id=35, shape=(), dtype=float32, numpy=2.718>
t.numpy()
## reutrn the value of 2.718
```



### Conflicting type

```python
try:
    tf.constant(1) + tf.constant(1.0)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
### cannot compute AddV2 as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:AddV2] name: add/
```



```python
try:
    tf.constant(1.0, dtype=tf.float64) + tf.constant(1.0)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
###cannot compute AddV2 as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:AddV2] name: add/
```



```python
t = tf.constant(1.0, dtype=tf.float64)
tf.cast(t, tf.float32) + tf.constant(1.0)
### This is correct
```



### String

```python
t = tf.constant("café")
### <tf.Tensor: id=44, shape=(), dtype=string, numpy=b'caf\xc3\xa9'>
tf.strings.length(t)
###<tf.Tensor: id=45, shape=(), dtype=int32, numpy=5>
tf.strings.length(t, unit="UTF8_CHAR")
###<tf.Tensor: id=46, shape=(), dtype=int32, numpy=4>
tf.strings.unicode_decode(t, "UTF8")
###<tf.Tensor: id=50, shape=(4,), dtype=int32, numpy=array([ 99,  97, 102, 233])>
```



### String array

```python
t = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
tf.strings.length(t, unit="UTF8_CHAR")
###<tf.Tensor: id=352, shape=(4,), dtype=int32, numpy=array([4, 6, 5, 2])>
r = tf.strings.unicode_decode(t, "UTF8")
###<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232], [21654, 21857]]>

```



### Ragged tensors

In most case, uneven distribution of data will be loaded in to tensor with uniform array. Which means you have to perform padding or clipping in some cases. 

Ragged tensor is the solution to this situation. Ragged tensor stores extendable array, which can easily store data with different length. Here are some application examples

- Changeable length feature list like actor names in films
- Multiple changeable length sequence like sentence or video clips
- Data with different layers like(letter,words,sentence)
- Structures certain area like buffer. 

Let take a look at a block of code

```python
speech = tf.ragged.constant(
    [['All', 'the', 'world', 'is', 'a', 'stage'],
    ['And', 'all', 'the', 'men', 'and', 'women', 'merely', 'players'],
    ['They', 'have', 'their', 'exits', 'and', 'their', 'entrances']])
print(speech)
###Expected output is
###<tf.RaggedTensor [['All', 'the', 'world', 'is', 'a', 'stage'], ['And', 'all', 'the', 'men', 'and', 'women', 'merely', 'players'],  ['They', 'have', 'their', 'exits', 'and', 'their', 'entrances']]>
```



Normally, the traditional operation of tensor can also be applied to ragged tensors.

tf.ragged also defines a lot of operations like tf.ragged.map_flat_values which can be used to convert certain value in ragged tensor with high efficiency while maintain its shape.

```
> print tf.ragged.map_flat_values(tf.strings.regex_replace,speech, pattern="([aeiouAEIOU])", rewrite=r"{\1}")
<RaggedTensor [['{A}ll', 'th{e}', 'w{o}rld', '{i}s', '{a}', 'st{a}g{e}'], ['{A}nd', '{a}ll', 'th{e}', 'm{e}n', '{a}nd', 'w{o}m{e}n', 'm{e}r{e}ly', 'pl{a}y{e}rs'], ['Th{e}y', 'h{a}v{e}', 'th{e}{i}r', '{e}x{i}ts', '{a}nd', 'th{e}{i}r', '{e}ntr{a}nc{e}s']]>
```

Here are two pictures that explain the difference between ragged tensor and sparse tensor.

![](img\sparseTensor.png)

![](img\RaggedTensor.png)

### Sparse tensor

Construct a sparse tensor

```python
SparseTensor(indices,values,dense_shape)
```

The indices is a 2D tensor with shape(n,ndims) which mark the nonzero position

````
indices = [[1,3],[2,4]]
````

means that at [1,3] and[2,4] position has nonzero elements which n is the number of nonzero elements and ndims is the dimension of this sparse tensor.

value is a N dimensional 1D vector which inputs the corresponding value in indice

dense_shape is a ndims dimensional 1D tensor which represent the shape of sparse tensor.

```python
tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
>>
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
```

Convert sparse tensor to normal tensor

```python
tf.sparse_to_dense(
sparse_indices,
output_shape,
sparse_values,
default_value=0,
validate_indices=True,
name=None
)
```

Sparse_indices are the position of those nonzero elements

- if sparse_indices is real number, then the matrix is 1 dimensional matrix , pointed to 1 certain element.
- if sparse_indices is a vector, then the matrix is 1 dimensional matrix , pointed to several elements.
- if sparse_indices is a 2 dimensional matrix then the matrix is multi-dimensional matrix and it is pointed to several elements in this matrix.

output_shape is the dimension of output matrix

sparse_value is the corresponding value of sparse_indices.

default_value is the value that will be assign to those element not in sparse_indices.

#### Sparse Tensor and Sparse tensor value

They shared same parameters.

- Use sparse tensor in computational graph 
- Use sparse tensor value in feed data.

Here is a block of codes that helps you understand sparse tensor better:

```python
s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
tf.sparse.sparse_dense_matmul(s, s4)
"""
Expected:<tf.Tensor: id=372, shape=(3, 2), dtype=float32, numpy=
array([[ 30.,  40.],
       [ 20.,  40.],
       [210., 240.]], dtype=float32)>
"""
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],
                     values=[1., 2.],
                     dense_shape=[3, 4])
print(s5)
"""
SparseTensor(indices=tf.Tensor(
[[0 2]
 [0 1]], shape=(2, 2), dtype=int64), values=tf.Tensor([1. 2.], shape=(2,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
"""
s6 = tf.sparse.reorder(s5)
tf.sparse.to_dense(s6)
"""
<tf.Tensor: id=381, shape=(3, 4), dtype=float32, numpy=
array([[0., 2., 1., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]], dtype=float32)>
"""
```



## Variable

```python
#Initilize variable
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
"""
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
"""
#get the value of variable
v.value()
"""<tf.Tensor: id=391, shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>

"""
#get the numpy array of this variable
v.numpy()
"""
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
"""

#update variable
v.assign(2 * v)
"""
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[ 2.,  4.,  6.],
       [ 8., 10., 12.]], dtype=float32)>
"""
#update certain element in variable
v[0, 1].assign(42)
"""
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[ 2., 42.,  6.],
       [ 8., 10., 12.]], dtype=float32)>
"""
#update vector in variable
v[1].assign([7., 8., 9.])
"""
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[ 2., 42.,  6.],
       [ 7.,  8.,  9.]], dtype=float32)>
"""
### we cannot directly update the value via traditional indexing methdo
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
"""
'ResourceVariable' object does not support item assignment
"""
sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                indices=[1, 0])
v.scatter_update(sparse_delta)
"""
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[4., 5., 6.],
       [1., 2., 3.]], dtype=float32)>
"""
v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.])
"""
<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
array([[100.,   5.,   6.],
       [  1.,   2., 200.]], dtype=float32)>
"""
```



### Devices

You can use tf.device() to make model to run on certain device like CPU or GPU or which of them.

```python

with tf.device('/gpu:1'):
    v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
    v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
    sumV12 = v1 + v2
 
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print sess.run(sumV12)
```

```python
#check module is run on which device
with tf.device("/cpu:0"):
    t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
t.device
```



## Some Basic low level customization method

#### Customize layer without weight

 Some layers have no weights, such as `keras.layers.Flatten` or `keras.layers.ReLU`. If you want to create a custom layer without any weights, the simplest option is to create a `keras.layers.Lambda` 

```python
### create a softplus layer
my_softplus = keras.layers.Lambda(lambda X: tf.nn.softplus(X))
my_softplus([-10., -5., 0., 5., 10.])
## use this layer in model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
    my_softplus
])
## Alternative way
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1, activation=my_softplus)
#   A few alternatives...
#   keras.layers.Dense(1, activation=tf.function(lambda X: my_softplus(X)))
#   keras.layers.Dense(1, activation="softplus")
#   keras.layers.Dense(1, activation=keras.activations.softplus)
#   keras.layers.Dense(1), keras.layers.Activation("softplus")
])
```



#### Customize layer with weight and bias

- The constructor

   

  ```
  __init__()
  ```

  :

  - It must have all your layer's hyperparameters as arguments, and save them to instance variables. You will need the number of `units` and the optional `activation` function. To support all kinds of activation functions (strings or functions), simply create a `keras.layers.Activation` passing it the `activation` argument.
  - The `**kwargs` argument must be passed to the base class's constructor (`super().__init__()`) so your class can support the `input_shape` argument, and more.

- The

   

  ```
  build()
  ```

   

  method:

  - The `build()` method will be called automatically by Keras when it knows the shape of the inputs. Note that the argument should really be called `batch_input_shape` since it includes the batch size.
  - You must call `self.add_weight()` for each weight you want to create, specifying its `name`, `shape` (which often depends on the `input_shape`), how to initialize it, and whether or not it is `trainable`. You need two weights: the `kernel` (connection weights) and the `biases`. The kernel must be initialized randomly. The biases are usually initialized with zeros. **Note**: you can find many initializers in `keras.initializers`.
  - Do not forget to call `super().build()`, so Keras knows that the model has been built.
  - Note: you could create the weights in the constructor, but it is preferable to create them in the `build()` method, because users of your class may not always know the `input_shape` when creating the model. The first time the model is used on some actual data, the `build()` method will automatically be called with the actual `input_shape`.

- The

   

  ```
  call()
  ```

   

  method:

  - This is where to code your layer's actual computations. As before, you can use TensorFlow operations directly, or use `keras.backend` operations if you want the layer to be portable to other Keras implementations.

- The

   

  ```
  compute_output_shape()
  ```

   

  method:

  - You do not need to implement this method when using tf.keras, as the `Layer` class provides a good implementation.
  - However, if want to port your code to another Keras implementation (such as keras-team), and if the output shape is different from the input shape, then you need to implement this method. Note that the input shape is actually the batch input shape, and the ouptut shape must be the batch output shape.

Let's take a look at an example

```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(MyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.biases = self.add_weight(name='bias', 
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)
        super(MyDense, self).build(input_shape)

    def call(self, X):
        return self.activation(X @ self.kernel + self.biases)
    
model = keras.models.Sequential([
    MyDense(30, activation="relu", input_shape=X_train.shape[1:]),
    MyDense(1)
    ])
```



#### Customized loss function

 Create an `my_mse()` function with two arguments: the true labels `y_true` and the model predictions `y_pred`. Make it return the mean squared error using TensorFlow operations. Note that you could write your own custom metrics in exactly the same way. **Tip**: recall that the MSE is the mean of the squares of prediction errors, which are the differences between the predictions and the labels, so you will need to use `tf.reduce_mean()` and `tf.square()`. 

```python
def my_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))
```

```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss=my_mse, optimizer=keras.optimizers.SGD(lr=1e-3))

```

```python
def my_portable_mse(y_true, y_pred):
    K = keras.backend
    return K.mean(K.square(y_pred - y_true))
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss=my_portable_mse,
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["mean_squared_error"])
model.fit(X_train_scaled, y_train, epochs=10,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
```



#### Customize function

Lets create Scaled Exponential linear unit: 

1. Take a quick review of the SELU function:

   $\text{SELU}(x) = \lambda    \begin{cases}    \mbox{$x$} & \mbox{if } x > 0\\    \mbox{$\alpha e^x-\alpha$} & \mbox{if } x \leq 0    \end{cases}$

2. Let implement this in tensorFlow

   ```python
   def scaled_elu(z, scale=1.0, alpha=1.0):
       is_positive = tf.greater_equal(z, 0.0)
       return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
   ```



### Function Graph

 The function's graph is represented on the following diagram. Call the graph's `get_operations()` method to get the list of operations. Each operation has an `inputs` attribute that returns an iterator over its input tensors (these are symbolic: contrary to tensors we have used up to now, they have no value). It also has an `outputs` attribute that returns the list of output tensors. Each tensor has an `op` attribute that returns the operation it comes from. Try navigating through the graph using these methods and attributes. 

![](img\cube_graph.png)

 Each operation has a default name, such as `"pow"` (you can override it by setting the `name` attribute when you call the operation). In case of a name conflict, TensorFlow adds an underscore and anindex to make the name unique (e.g. `"pow_1"`). Moreover, each tensor has the same name as the operation that outputs it, followed by a colon `:` and the tensor's `index` (e.g., `"pow:0"`). Most operations have a single output tensor, so most tensors have a name that ends with `:0`. Try using `get_operation_by_name()` and `get_tensor_by_name()` to access any op and tensor you wish. 



 Get the concrete function's `function_def`, and look at its `signature`. This shows the names and types of the nodes in the graph that correspond to the function's inputs and outputs. This will come in handy when you deploy models to TensorFlow Serving or Google Cloud ML Engine. 

```python
cube_func_int32.graph.as_graph_def()
```

