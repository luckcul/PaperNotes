# Convolutional Neural Networks for Sentence Classification

论文在预训练词向量基础上采用CNN对句子级别的文本进行分类。结果表明了无监督的预训练词向量(eg:word2vec)对于NLP任务有很大帮助。

## Model

![Model Architecture](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1519647660/paper/textcnn.png)

在如图的模型结构中，每一个词是一个k维的词向量x1，将句子中词向量拼在一起$x_{1:n}=x_1\oplus x_2 \oplus ... \oplus x_n$ ，对应了图中的第一层。这里画了两个channel，表示两个句子表示，一个channel是static,一个是non-static，表示是否随着网络训练而改变更新。

每一个filter的大小为h*k， h是窗口大小。第二层是卷积层，可以采用不同类型(不同窗口大小)的filter，同一个filter作用于不同的输入channel。
$$
c_i = f(w \cdot x_{i:i+h-1} + b) \\
c = [c_1, c_i,...,c_{n-h+1}]
$$
在第三层，进行`max-overtime pooling`即`global max-pooling`. $\hat c = max\{c\}$ ，主要作用是捕捉最重要的特征。在这里一个filter(一个输出channel)抽取一个特征。然后全连接到最后一层softmax层。

## Tricks

### dropout

dropout应用于倒数第二层。如果倒数第二层为$z=[\hat c_1, ..., \hat c_m]$ ，原本为
$$
y=w\cdot z+b\\
变为 \ \
y=w\cdot(z \circ r)+b
$$
r是m维大小，其中元素p概率为1。梯度传播时，只传播那些1的对应元素。在测试时，$\hat w = pw$

### l2-norm

也是作用在全连接层。$||w||_2 = s ,\ if \ ||w||_2 > s$ 

## Experiment

采用了7个文本分类数据集。

### 超参数设置

* 激活函数：relu
* filter windows：3， 4， 5 with 100 feature maps
* dropout rate: 0.5
* l2 constraint: 3
* mini-batch size : 50
* pre-trained word2vec dimension: 300

### 实验对比

* CNN-rand:不预训练词向量，随机初始化
* CNN-static:使用w2v预训练初始化，并保持不改变
* CNN-non-static:随着bp而fine-tuned
* CNN-multichannel:结合前两个，看做是两个channel，一动一不动

![exp](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1519650949/paper/textcnn-exp.png)

## Discuss

* 结果表明，这些预训练的词向量具有universal 特征抽取，可以跨数据集利用。
* 对于两个channel中词向量的变化(对比相近词，通过对比consine距离)。non-static中的词向量更贴切与任务。同时对于一些UNK的词也能学出一些意义的表示。

![](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1519651489/paper/textcnn-wv.png)

* Dropout能增加2%-4%的相对performance
* 对于不在w2v词表中的词，初始化为U[-a, a]，a是预训练词向量里的方差
* word2vec比其他词向量训练的效果要好

## Reference

* [开源实现](https://github.com/dennybritz/cnn-text-classification-tf)