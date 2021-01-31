# Graph Embedding之DeepWalk

Paper: DeepWalk: Online learning of social representations

论文14年发表，提出DEEPWALK使用随机游走的方式构造一种“特殊语言”，使用NLP模型学习节点表示。学习的向量表示与任务无关，下面是demo：

![https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610899180/paper/deepwalk/deepwalk_demo.png](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610899180/paper/deepwalk/deepwalk_demo.png)

本文的主要贡献是将NLP模型用以网络中社区结构的学习

## Random Walk

随机游走性能的两个优点：1）可以并行（使用多进程、多线程或多机器）；2）可以适应动态更新，只新增游走新增节点部分。

算法应用可行性分析中，语言中词频分布和随机游走生成的点击频度分布都服从幂律分布。通过skip-gram算法，拥有相似邻居的节点会得到相似的向量表示。

算法的超参数：r-每个节点的游走次数、d-生成向量的维度、t-每个节点游走的长度（句子长度）、w-skipgram窗口大小

![https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610903568/paper/deepwalk/deepwalk_algorithm.png](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610903568/paper/deepwalk/deepwalk_algorithm.png)

本文还介绍了使用的skip-gram算法，已经用以加速计算的Hierarchical Softmax技巧，具体是通过构造哈夫曼数来进行计算。将求softmax的复杂度从`O(|V|)`降为`O(log |V|)` 。

数据构建与算法训练示意

![https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610904100/paper/deepwalk/deepwalk_train.png](https://res.cloudinary.com/dyhtzpcxp/image/upload/v1610904100/paper/deepwalk/deepwalk_train.png)

由于数据满足幂律分布，这让我们可以使用异步的训练模式。具体用多线程来训练，没有使用锁机制来保护参数更新。

除了随机游走，对于某些可以自然生成序列样本（如网站中用户的页面访问）可以直接作为训练样本。这种方式不仅可以捕捉网络的结构，还可以捕捉访问的频度。

## Experiment

数据集使用了3类：BlogCatalog、Flickr、YouTube。

DeepWalk生成向量的使用方式：用LibLinear实现的one-vs-rest LR模型。具体超参数r=80、w=10、d=128

评价指标用了Micro-F1、Macro-F1，其中Micro-F1是全部样本在一起算TP、FP、FN，最终计算F1；Macro-F1是每一类分别计算F1，然后取平均。

## 源码

源码地址：[https://github.com/phanein/deepwalk](https://github.com/phanein/deepwalk)

### skip-gram

直接调用了`gensim` 的实现，使用了hierarchical soft的实现，默认`workers` 为机器cpu核数。

### random-walk

从给定起点出发（如果不给定，则随机一个），从当前节点，每次有一定概率（默认为0）返回初始点，否则从当前的邻居节点随机选择一个（包括已经访问过的）。如果当前节点没有下一个节点，或者长度足够则返回。

```python
def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]
```

构造样本库

```python
def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks
```

其余代码主要是处理样本比较大的情况，使用多进程、落盘、iterator等方式。