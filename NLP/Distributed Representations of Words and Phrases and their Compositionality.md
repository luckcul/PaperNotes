word2vec是自然语言处理中非常重要的基础。最近，读了一下相关论文，学习了相关的模型细节，并使用tensorflow实现了网络结构，对中文维基百科语料库进行了训练。相关论文及实现参考见文末。

# 关于词向量

在自然语言处理中，第一步就是要将自然语言数字化。所以，词向量的表示方式就很重要。最常见的有One-hot Representation和 Distributed Representation。

## One-hot Representation

这是一种很直接的表示方式，一个词向量中只有一个维度的值是1，其它的绝大多数是0。例如“dog”这个词的one-hot表示为`[0, 0, 1, 0, ......]`。 

这种表示方法得到的向量是非常稀疏的，维度非常大，会造成维度灾难。这种表示方法有一个很大的问题，如果两个词之间存在着关联，如果以one-hot方式表示的话，这部分关联信息就会丧失。模型几乎不能利用从“dog”学来的东西来处理“cat”，而distributed的表示方法能很好的解决这个问题。

## Distributed Representation

这种表示方法下，词向量的维度比较低，一般不会超过几百维。它的主要思想是，（句法和语义上）相近的词有相近的向量表示。这个相近的向量表示，可以用两个词向量之间的cosine夹角来度量。

这种表示方法也被成为word embedding。关于"distributed"的一种解释是，不像one-hot那样只在一个维度上有非零值。目前主要有两种训练模型：CBOW (Continuous Bag of Words) 和 Skip-gram model，两种模型比较相近。

# 训练模型与细节

## CBOW 模型

![](http://res.cloudinary.com/dyhtzpcxp/image/upload/c_scale,w_500/v1509196310/cbow_lbiet9.png)

模型的输入为某个词$w_t$前后的某些词，输出为要预测的$w_t$的输出y。假如窗口大小为2，则输入为$w _{t-1}, w _{t+1}$ ，这些输入是one-hot向量。每个词先通过矩阵W，映射到各自对应的向量；然后把这些各自得到的向量相加；最后通过一个softmax层，获得相应的概率。**其实，这个矩阵W，就是我们最终要得到的各个词的词向量的表示。**

## Skip-gram 模型

![](http://res.cloudinary.com/dyhtzpcxp/image/upload/c_scale,w_500/v1509197566/igSuE_au4ip3.png)

和上面的模型非常像，输入为某一个词$w_t$ ，输出为这个词前后词的概率。

如果一份文档由T个词组成，$w_1, w_2, ..., w_T$ ，那么skip-gram的目标函数就是最大化平均对数概率：
$$
\frac{1}{T} \sum_{t=1}^{T}\sum_{-c\leq j \leq c} \log p(w_{t+j } | w_t)
$$
其中，c表示中心词$w_t$前后各c个词。如果直接以softmax方式计算$p(w_O | w_I)$ 
$$
p(w_O|w_I) = \frac{\exp({v’_{w_O}}^\top v_{w_I})}{\sum_{w=1}^{W} \exp({v’_{w}}^\top v_{w_I})}
$$
其中，W是总词汇量，$v_{w}^{'}$和$v_w$ 分别是词w的输出和输入的向量表示，即分别是上图中两个矩阵$W_{V \times N}, W_{N \times V}^{'}$ 中的一行和一列。

然而，这样计算是不可行的，因为W实在太大了。

## 评价指标

论文中是采用analogical reasoning task来进行评价学到的词向量的好坏的。所谓的analogical reasoning task就是，比如给定`“Germany” : “Berlin” :: “France” : ?`。找到这样的词x，vec(x)和vec("Berlin") - vec("Germany") + vec("France")在cosine距离上最相近。

然而，感觉这种评价指标有点片面，而且对于具体的问题可能这项指标的作用不大。可以根据具体的任务相关来评价它的好坏。

## 优化

### Hierarchical Softmax

上面已经指出，计算softmax的复杂度太高，而通过Hierarchical Softmax，可以将每次的复杂度O(W)降到O(log(W))。

具体是，最后的输出层是一个二叉树，Huffman树。Huffman树是根据Huffman编码来进行得到的，这儿就不具体讲Huffman编码的算法了。最后的到的Huffman树，每个词代表的节点是叶子节点，而且词频越高的节点，深度越低。这样正向传播的时候，每次选择左子树，或者右子树，直到到达目标叶子节点，得到目标概率值。

### Negative Sampling

这也是对softmax层的优化，negative sampling的效率和效果都比Hierarchical Softmax要好些。Hierarchical Softmax是准确的计算概率，而这种方法是一种近似计算。对出目标词外的其他词进行负采样，可以降低其他单词的权重。

重新定义了上面的$p(w_O | w_I)$ ：
$$
p(w_O | w_I) = \log \sigma({v’_{w_O}}^\top v_{w_I} )+ \sum_{i=1}^{k}\mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-{v’_{w_i}}^\top v_{w_I})]
$$
其中，$ P_n(w)$ 是词w周围的噪声分布，具体的选取是根据词频越高，被选取的几率越大。k为除词w之外，其他要取的数量。论文中建议对于大的训练集，取2-5，小的训练集，取5-20。

### Subsampling

这是为了解决一些高频词，如”in","the"等。它们提供的信息非常少，而且这些词在对应的词向量在训练时也不会发生显著变化。为了解决高频词和低频次之间的不平衡，每个词有一定概率被丢弃：
$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$
其中，f(wi)是词wi的出现频率,t是一个选定的阈值，通常为1e-5。

# 基于维基中文语料库的训练

采用的数据是[中文维基百科](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)的数据，进行了数据抽取，繁体转简体，分词。

基于tensorflow来进行训练，在计算loss时，用了`tf.nn.nce_loss`来进行计算，它的输入是隐藏层到输出层的权重矩阵和偏差，隐藏层矩阵，真实值label，词典大小和negative sampling中采样大小k值。

实际的训练效果，示例如下：

![](http://res.cloudinary.com/dyhtzpcxp/image/upload/v1509346849/2017-10-28_23-02-55%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_izqrvh.png)

利用t-SNE对出现频度最高的500个词的词向量进行降维，下面是降维到2维的结果。

![](http://res.cloudinary.com/dyhtzpcxp/image/upload/v1509449657/blog/show_.png)

从图中可以看出，一些国家名称聚在了一起；"国王"，"皇帝"，"主席"，"总统"聚在了一起；"作品"，"故事"，"小说"等比较相近。

# 参考

[1].Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. Distributed representations of words and phrases and their compositionality. (NIPS 2013)

[2].[word2vec 中的数学原理详解](http://www.cnblogs.com/peghoty/p/3857839.html)

[3]. [TensorFlow: Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)

[4].[Wikimedia Downloads](https://dumps.wikimedia.org/)



