# Sequence model

## Example of some sequence data and application

- Speech recognition: Output text based on the input digital signal	
- Music generation:  Generate sheet music 
- Sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Name entity recognition



![](img\SequenceData.png)



## Math Symbol in sequence data

- Input X: "Harry Potter and Herminone Granger invented a new spell.” (Input as a sequence) $x^{<t>}$ is the $t\ th$ item in the input X
- Output y: like"1 1 0 1 1 0 0 0 0" (Human name location) and $y^{<t>}$ is the t th item in the output.
- $T_x$ is the length of input x
- $T_y$ is the length of output y
- $x^{(i)<t>}$ is the t symbol for i example
- Using dictionary to represent every word in the input sequence like one-hot vector.



## Problem  with standard network

![](img\RNN0.png)

- Inputs, outputs can be different lengths in different examples.
- Doesn't share feature learned across different positions of text.

## Recurrent Neural Networks

Recurrent neural network will take the activation from previous layer and apply into the computation of next layer.

We will need to create a $a^{<0>}$ for the initial state input, normally this $a^{<0>}$ is a zero vector, some researcher will used random initialization on this vector.

Recurrent Neural Networks scan through data from left to right.  Parameters were shared across each time steps.

![](img\RNN1.png)

- $W_{ax}$ governing the connection from input $x^{<t>} $ to hidden layers, each time steps use the same $W_{ax}$ 
- $W_{aa}$ governing the connection from activation to hidden layer
- $W_{ya}$ governing the connection from activation to output $y^{<t>}$

This Recurrent neural network has a problem that it can only use previous activation to calculate the $y^{<t>}$ . Bidirectional RNN will solve this problem.

### Forward propagation

![](img\RNN2.png)

- Initialize the activation vector $a^{<0>}=\vec0$
- $a^{<1>} = g(W_{aa}a^{<0>}+W_{ax}x^{<1>}+b_a)$ ; Normally, we will use tanh or ReLU as activation function. (Tanh function will cause vanishing gradient problem, we can solve it with other method.)
- $\hat y^{<1>}=g(W_{ya}a^{<1>}+b_y)$  ; For binary classification problem, we use sigmoid activation function, if that multi-classification problem, then we used softmax activation function)
- Note that $W_{ax}$ means it will required a variable $a$ and $W$ will be multiplied by $x$

Then we simplified the equations above:

- $a^{<t>}=g(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a)= g(W_{a}[a^{<t-1>},x^{<t>}]+b_y)$
- $\hat y^{<t>}=g(W_{ya}a^{<t>}+b_y)$
- Notation:  $W_a=[W_{aa}:W_{ax}]$, if $a^{<t-1>}$ is a 100 dimensional vector and $x^{<t>} $is 10000 dimensional vector. Then $W_{aa}$ is (100,100) dimensional matrix and $W_{ax}$ is (100,10000) matrix. We stack these two matrix together and get $W_a $  as (100,10100) matrix.
- $[a,x] = \begin{bmatrix}a^{<t-1>}\\x^{<t>}\end{bmatrix}$ , is a(10100) dimensional matrix.

![](img\RNN3.png)

## Back propagation for RNN

In the forward propagation there are four parameters: $W_a,b_a,W_y,b_y$

Back propagation and gradient descent will optimize these four parameters.

For backward propagation in RNN, we define new loss function:

- $L^{<t>}(\hat y^{<t>},y^{}<t>)= -y^{<t>}log \hat y^{<t>}-(1-y^{<t>})log(1-\hat y^{<t>})$ this is a just a normal logistic loss or we called cross entropy loss 
- $L(\hat y,y)=\sum^{t}_{t=1}L^{<t>}(\hat y^{<t>},y^{<t>})$

![](img\BackProp_RNN.png)

As we can see in the picture above, the direction of back propagation goes back through each time steps so we called this back propagation through time.

## Different type of RNN

- Many to many($T_x=T_y$) In this case the input and output have the same length.

![](img\RNN-MTM0.png)

- One to many : Like music generation,  we input a genre of music or a void value , and we get output of music sequence.

![](img\RNN-OTM.png)

- Many to One:  Like emotion classification problem, we  input a sequence and output only one or two value

![](img\RNN-MTO.png)

- One to One

![](img\RNN-OTO.png)

- Many to Many($T_x\neq T_y$) : Like machine translation

![](img\RNN-MTM1.png)

## How to build language model for RNN

### What is language modelling?

#### Speech recognition

- The apple and pair salad.
- The apple and pear salad.
- p(The apple and pair salad) = $3.2 \times 10^{-13}$
- p(The apple and pair salad) = $5.2 \times 10^{-10}$

### Language modelling with an RNN

- Training set: Large corpus of English text.
- Tokenize: Translate the entire sentence to one-hot vector.
- For those token that does not present in dictionary, we mark it as UNK
- First Step: Use zero vector for first prediction and output $\hat y^{<1>}$  prediction.
- Second step: Use previous output to predict next output
- Train the network, using softmax loss and update parameters and increase accuracy.

![](img\RNN-languagemodel.png)

## Sample Novel sequence 

After we finished the training of an sequence model, we want to know what it actually learned, there is an informal way of doing this called sample novel sequence.

For a sequence model, it will output the probability of word in certain sequence, what we want to do is to do sampling on these model and generate a new sentence.

For a trained RNN model:

- Input $x^{<1>}=0,a^{<0>}=0$ for the first time step and get all the possibility of it's outputs $\hat y^{<1>}$  according to softmax,  then we do random sampling and get the $\hat y^{<1>}$
- We used the output for previous time step $\hat y^{<1>}$ as the input and repeat this process.
- If there is an EOS in the dictionary, we end our generation once EOS was chose. If not, we can manually set the step number.

![](img\Rnn-novelsample0.png)

We can also create language model based on character instead of word.

![](img\Rnn-novelsample1.png)

But this language model based on characters cost more computational power and it does not performed as good as word based model on capturing the relation between previous steps and following steps.



## Vanishing gradient problem with RNNS

RNN does have a vanishing gradient problems:

- The cat, which .... , was full;
- The cats, which ......, were full;

In these two sentence, cat is corresponding to was, and cats is corresponding to were. There are long-term dependencies in a sentence but not many model can capture this type of relationship. 

The backpropagation process sometimes cannot pass the influence to early layer which means it cannot remember the word from very early layers. 

![](img\RNN-vanish.png)

Vanishing gradient problem is one of the most important problem for RNN. We might also encounter exploding gradient problem, but this problem can be found easily and can be fixed through gradient clipping.





## GRU Gated Recurrent Unit

Gate Recurrent Unit change the structure of RNN hidden layer so it can capture the relation and solve the vanishing gradient problem.

RNN unit:

For a RNN unit in a single time step, we can calculate the $a^{<t>}$ with this equation:

$a^{<t>}=g(W_a[a^{<t-1>},x^{<t>}]+b_a)$

![](img\RNN_UNIT.png)

## GRU(simplified)

For GRU we have a memory cell $c$.

- $c^{<t>}=a^{<t>}$ , the output of this memory cell is the activation of time step t.
- $\tilde c^{<t>}=tanh(W_c[c^{<t-1>},x^{<t>}]+b_c)$ for each time steps we have a $\tilde c$ as a candidate to replace original $ c^{<t>}$
- $\Gamma_u = \sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)$ is a 0-1 value indicate if we should replace the $c^{<t>}$ with $\tilde c$  

- $c^{<t>} = \Gamma_u * \tilde c^{<t>}+(1-\Gamma_u)*c^{<t-1>}$ This will help solve the vanishing gradient problem since $\Gamma_u$ is so easily set to 0 so basically  $c^{<t>}= c^{<t-1>}$ .

- $c^{<t>},\Gamma_u ,\tilde c^{}$ have the same dimension.  The $*$ is a element wise operation.

  ![](img\GRU-simplified.png)

  

## Full GRU

- $\tilde c^{<t>}=tanh(W_c[\Gamma_r*c^{<t-1>},x^{<t>}]+b_c)$. For a full GRU we add another Gate $\Gamma_r$ 
- $\Gamma_u = \sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)$
- $\Gamma_r = \sigma(W_r[c^{<t-1>},x^{<t>}]+b_u)$  this gate tells you how relevant is $c^{<t-1>}$ to computing the next $\tilde c^{<t>}$ 
- $c^{<t>} = \Gamma_u * \tilde c^{<t>}+(1-\Gamma_u)*c^{<t-1>}$
- $c^{<t>} = a^{<t>}$

![](img\GRU-FULL.png)

## LSTM Long short term memory unit

The performance of LSTM is better than GRU at capturing deep relationship.

![](img\LSTM0.png)

In LSTM we have three Gate $\Gamma_u,\Gamma_f,\Gamma_o$ for update, forget and output.

- $\tilde c^{<t>}=tanh(W_c[a^{<t-1>},x^{<t>}]+b_c)$
- $\Gamma_u = \sigma(W_u[a^{<t-1>},x^{[<t>]}]+b_u)$
- $\Gamma_f = \sigma(W_f[a^{<t-1>},x^{<t>}]+b_f)$
- $\Gamma_o = \sigma(W_o[a^{<t-1>},x^{<t?}]+b_o)$
- $c^{<t>} = \Gamma_u * \tilde c^{<t>}+\Gamma_f*c^{<t-1>}$
- $a^{<t>} = \Gamma_o*tanhc^{<t>}$

![](img\LSTM1.png)



These gate value can not only based on $a^{<t-1>},x^{<t>}$, might also depend on the value of last memory cell $c^{<t> }$  which also called peephole connection.

 

## Bidirectional RNN (BRNN)

BRNN is a solution that allows RNN to consider the future information.

![](img\BRNN0.png)

The equation to compute y is different  $\hat y ^{<t>}=g(W_y[\overrightarrow{a}^{<t>},\overleftarrow{a}^{<t>}]+b_y)$

![](img\BRNN1.png)

If we use this implementation, we must have the entire sequence to have output prediction.

## Deep RNNs

Deep RNNs has multiple layer recurrent. It is hard to see a 100 layers RNN. Due to its sequence feature 3 layer could be really big network already.

![](img\DeepRNNs.png)

