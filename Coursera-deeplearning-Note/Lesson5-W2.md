#Natural language processing

## Word representation

Computer cannot recognize the word like human so we do representation for words so that computer can understand them.

### One-representationtion

We use a vector to represent a word which there all only "1" in vector and rest element are "0"s.

![](img\one-hot rep.png)

Shortcoming of one-hot: We isolate every word since their distance are the same and product are zero. Model cannot learning the relevance and similarity of each word.



### Featurized representation : Word embedding

Using different feature to represent different word like in following picture:

![](img\Featurized-rep.png)

This representation finely express the relevance and similarity between words, thus the performance of generalization will be better.

If we able to learning 300 dimensional vector we can do T-SNE algorithm we can find 2D visualization of these data.

![](img\Word-embedding.png)

## Use word embedding

Name entity recognition example:

![](img\name entitiy_example.png)

Assume that we have a relative small training set that does not contain durian and cultivator. It is hard to detect these name entity, but if we have a learning word embedding that told us Durian is a kind of fruit and cultivator is like farmer then we might detect these name entity from our small training set.



### Transfer learning and word embeddings:

1. Learn word embedding from large text corpus. (1-100billion words) or download pretrained embedding online.
2. Transfer embedding to new task with smaller training set.(100k words)
3. Optional: Continue to finetune the word embeddings with new data.



### Relation to face encoding(embedding)

For face recognition, we always encode face into different encodings as their unique representation, word embedding is kind of similar to face recognition.

![](img\Word_face.png)

But there are difference, for face recognition,we can randomly place  a face encoding into our network and output a encoding but for word embedding all the embeddings are mean to stay in a fix table and find the relationship between each other.



## Properties of word embeddings

### Analogies:

There is important feature of word embeddings: Analogies reasoning, we can do subtraction between each word embeddings and found their relation.

This concept help research to have better understanding to word embedding.

![](img\Analogies.png)

If we do subtraction on man and woman, king and queen , we will find out that their main difference is gender.

### Analogies using vectors

Calculating the similarity between words are actually finding the distance similarity  at each dimension for each word.

![](img\analogies1.png)

$e_{man}-e_{woman}\approx e_{king}-e_{?}$

For the equation above, we can use following equation to compute $e_?$:

$arg max sim(e_?,e_{king}-e_{man}+e_{woman})$

Similarity function:

- Cosine similarity : $sim(u,v) = \frac{u^Tv}{||u||_2||v||_2}$
- Euclidian distance: $||u-v||^2$



## Embedding matrix 

When we are going to learn a word embedding model, the essence of this process is to learning the embedding matrix $E$. When we learned this matrix, we can multiple this matrix with the one-hot vector corresponding to each word to get the word's embedding.

![](img\embedding matrix.png)

## Learning word embeddings

 In the early stage of learning word embeddings, people create complex algorithm but when time goes on people found that simple algorithm can also create great result with larger data set.

Early stage of learning algorithm: 

In the following example we are going to use previous word to predict last word.

- Multiply the one-hot vector with embedding matrix and get word embeddings.
- Have a fixed historical window (fix number previous word required by predicting next word) . Stack the word in the window into an input vector of the neural network.
- Then we use softmax layer to output the probability of each word.
- All the hidden layer and softmax layer have their parameter. Assumed that our corpus has 10000 words and each word embedding is 300 dimensional. Historical window has size of 4. Then we are going to input $300\times 4$ and output 10000
- Parameters of the model are embedding matrix, weight of hidden layer and softmax layer $w^{[1]},b^{[1]},w^{[2]},b^{[2]}$ 
- Then we used back propagation to perform gradient descent to train the maximum likelihood  function and predict next word.

In the training process, if the algorithm will try to best fit the training set, it make similar words to have similar feature vector and thus format the embedding matrix $E$

There are several method to select words to predict next word:

- Select several words before target word
- Select several words after target word
- Select one word before target word
- Select a random word near target word.

## Word2Vec

Skip-grams:

In the skip-gram model, we need to select Content and target to match and create a supervised learning problem. Content are not necessarily be a word closest to the target, it could be a random word nearby the target word within a range. This model is not meant to solve the supervised learning problem instead to learn a great word embedding model.

Model detail:

- Use a large corpus like: Vocab size = 10000K
- Create a basic supervised learning problem, find Context C and Target  and establish relationship.
- $o_c$(one-hot), $E$ embedding matrix, $e_c = E*o_c$ word embedding, $Softmax\ layer -\hat y$
- Softmax:$p(t|c) = \frac {e^{\Theta^T_te_c}}{\sum^{10000}_{i=1}e^{\Theta^T_je_c}}$,  and $\Theta_t$ is the parameter associated with output t.
- Loss function: $L(\hat y,y)=-\sum^{10000}_{i=1}y_i log \hat y_i$. This is the loss function for softmax layer when target y is a one-hot vector.
- Through back propagation and gradient descent we can have $E$ and softmax parameter.

#### Existed problem:

- Computational efficiency: In above softmax units we will do sum for entire vocab, this computation will generate high computational cost.
- Solution: Hierarchical softmax classifier(Binary tree classifier, each node is a binary classifier). The computation complexity is $log|v|$ . Normally the commonest word will be place on the top of the tree so this is a unbalanced tree.



## Negative Sampling

Skip-grams model will help us create a supervised learning task, mapping the context with target word and learn a good word embedding. 

Negative sampling is similar to Skip-gram model but with higher efficiency.

Learning task:

- Define a new learning task: predict if two words are context word and target word, if the word matched with context word then mark the target with 1 else 0.
- Choose different word with same context k times and mark the target and generate the training set.
- Tips: For small training set k= 5~20, for large training set k = 2~5.
- Then we learning mapping from $x-y$ which x is a pair of word and y is 0 or 1

Model:

In negative sampling model we use logistic regression model:

$P(y=1|c,t)=\sigma(\Theta^T_te_c)$

For each positive sample we have k negative samples. In the training process for each context word we will have k+1 corresponding classifier.

![](img\Negative sampling.png)

Comparing to Skip-gram model, negative sampling no longer need a  complex softmax layer with vocab size Time complexity. Instead we transfer this problem into vocab size binary classification problem since every problem has 1 positive example and k negative examples.



### How to choose negative examples:

After we select context we have 1 positive example and we still need k negative examples to train every classifier.

- Sampling based on the frequency: a, the ,of might show up more frequently than other words.
- Uniformly random choose negative example: Result is no representative.
- (Recommended): $P(w_i)=\frac{f(w_i)^{3/4}}{\sum^{10000}_{j=1}f(w_j)^{3/4}}$ this method is in between two method, that we choose neither frequency nor uniform distribution. We have 
- We sample the frequency of the word to the power of $\frac{3}{4}$, $f(w_i)$ is the frequency of a word

## GloVe Algorithm(Global vectors for word representation)

Glove is another model to compute the word embedding, it was used less common than skip-gram but came up with simpler structure.

### GloVe model:

In the GloVe model, we need to define a $X_{ij}$ to represent times of $i$ appears in the context $j$ .

Optimization:

$minimize\sum^{10000}_{i=1}\sum^{10000}_{j=1}f(X_{ij})(\Theta^T_ie_j+b_i+b_j-logX_{ij})^2$

- When $X_{ij}=0$, $log(X_{ij})$ has no meaning, so we add a weighted term $f(X_{ij})$ when $X_{ij}=0,f(X_{ij})=0$. $f(X_{ij})$ also balance the word with very low frequency and very high frequency.
-  $\Theta^T_i$ and $e_j$ are all learnable parameter. In this optimization, they are symmetric so we can initialize them identically. Then we use gradient descent to minimize the loss then used the average of them to be the word embedding. $e^{final}_w=\frac{e_w+\Theta_w}{2}$

### A note on the featurization view of word embedding

We cannot promise that very dimension of word embedding are interpretable. But we can assure that every dimension are relative to some feature we understand. Some of them might be the combination of some interpretable dimensions.

We explain this GloVe with linear algebra :

$\Theta^T_ie_j=\Theta^T_iA^TA^{-T}e_j=(A\Theta_i)^T(A^{-T}e_j)$

A term might create any dimension.

![](img\Featurezation-view of word.png)





## Sentiment classification

Sentiment classification is parse a sentence and check if this sentence like or hate the content it mentioned.

![](img\sentiment classification.png)

There is a problem about the sentiment classification is that we only have very small training set. But when we use the word embedding, it will improve the performance to a acceptable level.

### Average or sum model:

- Acquire a trained embedding matrix
- Get every word embedding and do average or sum on every word embedding
- Input to the softmax classifier and get output $\hat y$
- Problem: This model does not considered sequence of word. This might make wrong prediction due to large amount of positive word and compromise the effect of previous negative words.
- ![](img\Average_sum.png)

### RNN model:

- Acquire a trained embedding matrix E
- Get every word embedding and input to many-to-one RNN model;
- Through softmax classifier and get output $\hat y$
- Advantage: Sequence order are considered thus improve performance

![](img\RNN-Sentiment.png)

## Debiasing word embeddings

Some of current NLP model or embeddings will output result with gender or race bias. This reflect the fact the people used to have these biases in writing history. 

![](img\Bias.png)

### Solution:

- Identify bias direction like gender
  - Minus those gender related word with paired word and average them.
  - After we have the average, we can have one or more related dimensions and many unrelated dimensions.
- Neuturalize: For every word that is not definitional, project to get rid of bias.(For word like doctor, babysitter)
- Equalize pairs: Adjust the position like grandmother and grandfather to position like babysitter thus make babysitter become a neutral position.

![](img\Bias-RNN.png)