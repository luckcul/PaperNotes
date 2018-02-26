# Bag of Tricks for Efficient Text Classification

论文提出了，**fastText**用在文本分类。 作者Tomas Mikolov正是word2vector的作者，fastText型的结构也和w2v比较相似。本文的主要贡献是提出了fastText模型，它的效果表现和表现最优的一些模型相差较小，但是速度比较快，有数量级的差距。该模型并没有创新，速度是该模型的亮点。开源的[fastText](https://github.com/facebookresearch/fastText)用于文本表示，进行训练词向量和文本分类，本文主要介绍的是文本分类。

## Model Architecture

![Model Architecture](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1519644778/paper/fastText.png)

模型结构如上，和word2vec中的CBOW(continuous bag of words)模型结构相同。差别就是本文模型是有监督的，输入是文本的N-gram特征，输出是预测类别label。

## Tricks

### Hierarchical softmax

和word2vec中的采用的一样，目的是为了加速计算softmax正向和bp。回忆一下，此时的目标函数为：
$$
-log\ \sigma{({v'_{w_o}}^\top v_{w_I})} - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma (-{v'_{w_i}}^\top v_{w_I})]
$$


一般这个二叉树为Huffman树，这样极大减少了训练的时间。但是在预测的时候还是需要计算出最大的softmax值的，采用的是带剪枝的dfs，还有一些其他技巧。

关于为何没有采用w2v里面的Negative Sampling，原因是对于训练阶段是可以的，但是预测阶段，是和普通softmax效率是一样的，效率比较低。解释可以见[这儿](https://fasttext.cc/docs/en/faqs.html)

> However, negative sampling will still be very slow at test time, since the full softmax will be computed.

### N-gram features

使用词袋模型，有一个缺点，失去了词序这个信息。为了弥补这个不足，fasttext增加了n-gram特征。具体是把n-gram当成一个词，也进行embedding表示，模型中也把他们进行求和取均值。而在具体实现上，n-gram数量非常多，不能完全存下，具体上进行hash，有可能多个n-gram共享一个embedding vector。而单个词会单独享用一个vector.

## Experiments

主要进行了两类任务进行评测Sentiment analysis和Tag prediction。第一个任务主要是为了在一般的文本分类任务上，在效果和时间上来衡量fasttext。它的速度和其他模型有数量级的差距，在cpu下就可以有不错的速度和效果。

tag预测，采用的数据集中训练集有91,188,648个sample，tag有312,116个不同标签。对比的方法是Tagspace。