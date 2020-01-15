# Sequence model and attention

## Basic model

### Sequence to sequence model:

The most common sequence to sequence model is machine translation, assumed that we want to translate French to English:

- Input:$x^{<1>},x^{<2>},x^{<3>}...x^{<T_x>}$: $x^{<t>}$ is every French word in sentence
- Output:$y^{<1>},y^{<2>},y^{<3>}...y^{<T_x>}$: $y^{<t>}$ is every English word in output
- Network Structure: Many to many

If we have a large training set then we can get a machine translation model that works well. We have a encoding network to encoding the input French sentence and output the result to a decoding network to generate the English translation.

![](img\SequenceToSequence.png)

### Image to sequence model:

Image to sequence model is very similar to Sequence to sequence model.

- Input: Image
- Output: Sentence to describe the image
- Network structure: CNN to encode the image and RNN to decode the image and output corresponding caption

![](img\ImageCaption.png)





## Picking the most likely sentence

Machine translation as building a conditional language model:

Machine translation is quiet similar to the language model but there are some difference.

- In the language model, we can estimate probability of a word to generate new sentence. Language model can input a zero vector
- In the machine translation, there are encoding network and decoding network. The decoding network is quiet similar to language model. Machine translation model takes word or image as input. Comparing to normal language model, it is called conditional language model.

![](img\Conditional_language_model.png)

### Finding the most likely translation:

According to input French sentence, the model will output the probability of every translation.

![](img\MostLikely translation.png)

For every possible translation result, we do not do random sampling from the distribution, instead we find a condition that filter the max probability output. So when we design the machine translation model, designing a appropriate algorithm that find the most likely translation is very important.

The most common algorithm is "Beam search"

### Why we don't use greedy search:

Greed search: After we have the first word, greedy search will generate following output based on the first word.

But for machine translation, we need an algorithm that pick very output at once and maximize the total probability.

And for greedy search, it is impossible to compute every combination for the million or tens million size corpus.  So we used likely search to maximize the probability instead of use word. It cannot guarantee that we have the max probability but the performance is acceptable.

![](img\WhyNotGreedy.png)

## Beam search

### Beam search algorithm:

Taking French to English machine translation as example:

- Step 1: For our vocab, we input French sentence into our encoding network and get the encoding of sentence. Through a softmax layer, we compute probability of each word. By setting beam width  (for example 3) then we store words that have top3 probability. 

![](img\Beam_step1.png)

- Step 2: For every word we got from step 1, we compute the probability of the next word. And then multiply with probability of first steps and get the probability of a pair. Then we store the pairs with top3 probability

![](img\Beam_step2.png)

- Step3: Repeat Step 2 until we meet a end mark of the sentence.

![](img\Beam_step3.png)

## Refinements to beam search

#### Length normalization

For beam search, our target is to maximize the probability or output sentence.

$argmax\Pi^{T_y}_{t=1}P(y^{<t>}|x,y^{<1>}...,y^{<t-1>})$

But every item in this equation is smaller than 1 since they are probabilities. After several time of multiplication, it will become a very very small value. 

To prevent this situation, we change the equation to below:

$argmax_y\sum^{T_y}_{y=i}logP(y^{<t>}|x,y^{<1>}....,y^{<t-1>})$

This log will provide a relatively stable numerical result and this log function is a strictly monotonically increasing function maximizing P(y).

This equation for optimization will prefer short sentence since the longer the sentence the smaller the probability will be(Times of multiplication increased.)

To solve this problem we make another change to the equation:

$\frac{1}{T_y}\sum^{T_y}_{y=i}logP(y^{}|x,y^{<1>}....,y^{<t-1>})$

After we normalize the probability this will give us better output.

There is another soft approach:

$\frac{1}{T_y^\alpha}\sum^{T_y}_{y=i}logP(y^{}|x,y^{<1>}....,y^{<t-1>})$

By adding a $\alpha$ to normalization term might help with this problem. But this method is not yet justified. Normally set to 0.7

![](img\Beam-searchopt.png)

For Beam search:

- Beam size B: The bigger the B, more situation and possibility were considered, but the computational cost will increase as well. In normal situation, B =10, bigger B like100 or 1000 will need to consider the special application.

![](img\beam-searchopt1.png)



## Error analysis on beam search

 Beam search is a heuristic algorithm that only output approximate result. Error analysis is a great method to help  developer to check if there RNN went wrong or the beam search went wrong.

Example:

We used  French to English translation as example. We have our human translation(y*) and machine output(y^). And we are going to break the problem apart.

The RNN compute P(y*|x) and P(y^|x)

- Case1: $P(y*|x)>P(\hat y|x)$ which means that y* has higher(y|x) but our beam search algorithm chose $\hat y$. This means that our beam algorithm is wrong and needs modification.
- Case2:$P(y*|x)<P(\hat y|x)$ This means that our RNN output the wrong probability.

![](img\eror-analysis.png)

### Error analysis process

In the dev set, we check every sentence for $P(y*|x)andP(\hat y|x)$. Then we can divide the result into two case and check the fraction. Based on the fraction of fault result, we can plan which part we should put more effort into, RNN or beam search.

![](img\error_analysis1.png)



## Bleu score

For machine translation there are always more than one great answer, unlike image recognition which only has one correct answer. BLEU score is a method that helps developer to evaluate the result. BLEU stands for bilingual evaluation understudy.

BLEU score need at least one human translation reference. 

![](img\BLEU0.png)

In the picture above, the machine translation(MT) is clearly a bad output but since 'The' shows up seven times and "the" is in the reference. The precision will be very high. So we needs to modify this precision. The modified precision will shows that the count both the number of a score shows up in reference and MT.

### BLEU score on bigrams(Pair of words)

![](img\BLEU1.png)

The equation of computing BLEU for n-grams is:

$\frac{\sum_{n-gram\in \hat y}Countclip(n-gram)}{\sum_{n-gram\in\hat y}Count(n-gram)}$



### BLEU details

$p_n$= Bleu score on n-grams only

Combine Bleu score: $BP\ exp(\frac{1}{4}\sum^n_{n=1}=p_n)$

BP is called brevity penalty. 

BP=1 when the output is actually longer than input text.

BP=exp(1-MT_output_length)/(reference_output_length)

![](img\BLEU2.png)



## Attention model intuition

Attention model will help the model to do better on the long sentence.  When human do translation, we do not memory the entire sentence since memorize an entire sentence is very hard. So we actually do translation part by part. The performance of  encoding and decoding RNN will decrease after the length of sentence exceed a value. Just like the picture below. The attention will help make the RNN like human translation and translate the sentence part by part.

![](img\attention_model.png)



## Detail of attention model

We  use a Bi-direction RNN model as example and get corresponding English sentence. Every unit in RNN is LSTM or GRU. For bi-direction RNN and forward backward propagation, we can have the forward and backward activation at every timestep. We use "input" to represent these activation values. 

At the meantime we have a decoding network for English output. By taking attention weight we can have a sum that include context as input to the decoding network. Also, the activation value from previous timestep will also be an input of next timestep.

![](img\attention_model1.png)



Computing attention weight:

![](img\attention_model2.png)

For $a^{<t,t'>}$ we use a softmax function to make sure that the sum of probability  is one. There is also a small layer of network(normally one hidden layer) to compute the $e^{<t,t'>}$ according to $s^{<t-1>} \ and\ a^{<t'>}$ 

The problem with this attention model is that it has large time complexity $O(n^3)$

However, the high complexity is acceptable since the translation usually has short sentence.

### Attention model example:

- Change time format to formal format.
- Visualize $a^{<t,t'>}$

![](img\attention_model3.png)



## Speech recognition:

Speech recognition is a process that transfer a audio signal to corresponding text.

### Speech recognition problem:

In Speech recognition problem, we design a process to transfer audio signal to spectrum(Time/frequency map that mark different spectrum energy with different color). The spectrum will be input to algorithm to process.

In the early stage of speech recognition, linguists create phenomes to study. But with the advancement of current deep learning, this phenomes representation is no longer necessary. We can do transfer directly from audio signal to text with hand designed feature.

![](img\speechRecogntion.png)

The most common size of audio clip for current speech recognition application is 300 hours or 3000 hours. For more precise recognition model, over 10000 hours audio clips might be applied.



To build a speech recognition model, we can use the previous attention model we learned.

![](img\speechRecognition1.png)



There is also another method that improve the performance of speech recognition model call CTC loss.

### CTC loss function

CTC, Connectionist temporal classification

For speech recognition problem, the input audio signal usually are much larger than output text.(10s audio clip might have 1000+ input feature)

For this problem, we use CTC loss function. CTC loss function allows our RNN model to output repeated character and blank separator. After we output the sequence we collapse the repeated character between blank separator. 

![](img\CTC.png)



## Trigger word detection system

Example: Amazon Echo, Baidu DuerOS, Apple Siri, Google Home.

Trigger word detection algorithm.

Take a clip and generate a spectrum based on the input audio clip, and set the output of our training set equals to 0 for the first output after trigger word equal to 1 for those part after trigger words.

![](img\trigger word detection.png)

There is a problem with this algorithm that 0 and 1 label are unevenly distributed.  We can set output value equal 1 for multiple times after we hear the trigger word and delay the time it was reset to 0.