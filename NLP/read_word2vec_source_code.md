# read word2vec source code

* 因为对sigmoid的精度要求不是很高，而且要重复用到。所以sigmoid 进行打表。

```c
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
```



* hash 表部分。。大小保持不变3e7大小。。如果现有的词超过了70%，就删去出现频度<= 1的。下次再超过就删去<=2的……则最后显示的vocab不是真正所有的，只是一个大概。`words in train file` 的数量不包括词表外的词。
* `LearnVocabFromTrainFile()` 得到`vocab[]`和`vocab_hash[]`，最终按照#word >= min_count 来筛选。数量`vocab_size` 和 `train_words`。 
* `CreateBinaryTree` 从有序数组里面构造Huffman tree 的写法很优美。为什么要记录路径的point ---为了求hierarchical softmax而记录。
* `InitNet()`做的事，， 开辟syn0 input vectors, 对于hs or negative 开辟syn1，并创建huffman tree,并写入到`vocab[].point and code`。output vector `syn1` 初始化为0,input vector `syn0`初始化为`[-0.5/layer_size, 0.5/layer_size]`

```c
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
```



* `InitUnigramTable()` 是初始化table数组。按照采样概率，在这个1e8大小的table上，每个单词的出现的次数=被采样的概率*1e8。
  * 使用的是unigram distribution, 3/4是经验值

$$
P(w_i)=\frac{f(w_i)^{3/4}}{\sum_{j=0}^{n}f(w_j)^{3/4}}
$$

* alpha 学习速率的变化。初始0.025 `starting_alpha = alpha = 0.025`  `alpha`更新 

```c
alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001
```

* `iter` 默认5次。
* 下采样

```c
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
```

这和论文是不一样的
$$
P(w_i) = \sqrt {\frac{t}{f(w_i)}} + \frac{t}{f(w_i)}
$$

* 并不是在`windows`里面的所有词来做训练， 而是random b = [0,windows)。然后`for (a = b; a < window * 2 + 1 - b; a++)`。 
* 在cbow模型中，并没有把梯度的1/cw更新到每个input vector。而是`for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];`
* `word2vec.c` 还提供了了通过k-mean聚成预先设定好的classes个类，对每个词聚类。