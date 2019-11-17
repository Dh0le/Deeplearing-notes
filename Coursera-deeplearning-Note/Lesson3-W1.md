# Machine Learning Strategy

## Why machine learning strategy

- There are many ways to optimize a machine learning model, how to chose the most effective way becomes more and more important in the iteration cycle. 
- There are following ideas that might effect machine learning process.
  - Collecting more data
  - Collecting more diverse training set
  - Train algorithm longer with gradient descent
  - Try Adam?
  - Try bigger network
  - Try smaller network
  - Drop out Add L2 regularization
  - Change number of hidden unit
  - Change activation function



## Chain of assumptions in ML

1. Fit training set well on cost function (comparing to human level function)
2. Fit dev set well on cost function
3. Fit test set well on cost function
4. Performs well in real world.

We should have a separate tune method for each standard--Orthogonal control 

1. Bigger network, Adam for 1
2. Regularization for 2
3. Get a bigger dev set for 3
4. Change the dev set distribution or cost function.

- early stopping might not be a great method in this , since it effect two things, and it is harder to analyze.

## Using a single number evaluation metric

For classifier:

- Precision: Of example that the classifier recognized as positive is truly positive.
- $Precision =\frac{True \ positive}{Num\ of \ predicted\ positive} \times 100 \%$
- $Precision = \frac{True \ positive}{True\ positive+ False\ positive}$
- Recall: What  % of actual positive are marked as positive.
- $Recall = \frac{True\ positive}{Num \ of \ acutally \ positive} \times 100\%$
- $Recall =  \frac{True \ positive}{True\ positive +False \ negative}$
- There is a trade off between both.
- With two standard it is hard to compare.
- F1 score = "Average of P and R"
- $\frac{2}{\frac{1}{P}+\frac{1}{R}}$, harmonic mean



## Setting up Train/dev/test set distribution

- Guideline:
- Choose the dev set and test set to reflect data you expect to get in the future and consider important to do well on.

## Size of the dev and test set

Old way of splitting data

- 70% for training set and 30% for test set
- 60% for training set 20% for dev set % 20% for test set
- These splitting conventions are still reasonable for total data like under 10,000

Modern way:

- Since modern total data set sizes are getting bigger and bigger, like 1 million examples
- Then it is reasonable to use 98% for training set and 1% for dev 1%for test set

Size of test set:

- Set your test set to be big enough to give high confidence in the overall performance of your system.
- It could be less than 30% of the over all dataset. 

Not having a test set is okay when it does not need high confidence on the overall system final performance.

## When to change dev/test sets and metrics

Sometime we need to add weight term into metric so we can make the metrics more appropriately evaluate for desired outcome

If you are unsatisfied with old error metric, stop using them and starts to modified with them.

### Orthogonalization for cat pictures :"Anti porn"

1. How to define a metric
2. Worried separately about how to do well on this metric.

If doing well on metric +dev/test set does not correspond to  doing well on your application, change metric and/or dev/test set.

## Why human level performance

### Comparing to human-level performance

- the accuracy of a certain algorithm will surpass **human level performance** and stay in very slow increase speed and stay below the **Bayes optimal error**.

- Bayes optimal error is the very best theoretical function for mapping from x to y. That can never be surpassed.

### The reason why model slows down after surpass human level performance

1. Sometimes it is because the distance between human level performance and Bayes optimal error is very close. So there are not much space for model to improve.
2. So long as your model are worse than human level performance, there are many tools that can efficiently improve performance but once the model surpass the human level performance, those tools are no longer useful.
3. Human are quite good at a lot of tasks so long as ML model is worse than human you can:
   1. Get labeled data from human.
   2. Gain insight from manual error analysis
   3. Better analysis of bias/variance.

## Avoidable bias

If there is a huge gap between algorithm performance and human performance, then we might need to focus on bias. Maybe train longer, bigger network. If the model performance has very little game between human level performance, then we might need to focus on the gap between training set and dev set.

Term: **Avoidable bias** is the error between Bayes error and training error.  Which means there is a space that developer should not get below.



To decrease the avoidable bias

- Train bigger network
- Train longer/better optimization algorithm
- NN architecture/hyperparameters search RNN/CNN

To decrease the variance 

- More date
- Regularization
- Dropout









