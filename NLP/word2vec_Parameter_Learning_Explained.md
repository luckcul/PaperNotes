# word2vec Parameter Learning Explained

之前读过word2vec的提出论文《Distributed Representations of Words and Phrases and their Compositionality》，并在tensorflow下进行实现，用维基数据集进行了训练。但是感觉一些细节理解和训练过程的含义，有一点模糊的感觉，所以读了该篇论文。它比较详细的解释了CBOW和Skip-Gram两种模型的训练过程，包括参数学习更新，以及直观理解。接下来准备阅读一下word2vec的c源码，进一步理解。

本篇论文主要解释了CBOW(Continuous Bag-of-Word)模型和Skip-Gram 模型，分别介绍了模型网络结构，损失函数，梯度计算。还有两个优化Hierarchical Softmax和Negative Sampling提高训练效率。并提供了一个可视化的训练过程演示[wevi](https://ronxin.github.io/wevi/)。

## 直观解释

对于参数更新，给出了直观的解释。以CBOW为例，给定上下文的词，预测target word。
$$
p(w_j|w_I) = \frac{exp({v'_{w_{j}}}^{T}v_{w_{I}} )}{\sum_{j'=1}^{V}exp({v'_{w_{j'}}}^{T}v_{w_{I}})}
$$
用output vector $ v'_{w_{j}}$ 和input vector $v_{w_{I}}$ 的內积表示他们的距离。损失函数其实就是交叉熵损失，输出层的输入值的梯度就是预测值减去真实值。

### hidden->output weights update

$$
{\textbf{v}'_{w_{j}}}^{(new)}={\textbf{v}'_{w_{j}}}^{(old)} - \eta \cdot e_{j} \cdot \textbf{h} \ \ \ \ \ \ for\ j = 1, 2, \cdots, V
$$

这是output vector的更新，$e_{j}$ 就是上面说的输入值梯度，h是hidden层向量。

如果$y_i > t_i$， 即真实值为ti=0，但是预测概率较高overestimation，output vector $v'_{w_{j}}$就会减去一定比例的hidden vector h，这样$v'_{w_{j}}$就会远离$v_{w_{I}}$。 如果$y_j < t_j$， 即underestimating， 这时$w_j = w_O$， 加上一定比例的h，使得$v'_{w_{o}}$距离$v_{w_I}$更近。 

这里的更近更远是用inner product来衡量的，一个向量a减去一定比例的向量b，a和b的点积会更小，“距离”会更远。

### input->hidden weights update

$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{V}\frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial h_i} := EH_i
$$

假设只有一个词输入,
$$
{\textbf{v}'_{w_{I}}}^{(new)}={\textbf{v}'_{w_{I}}}^{(old)} - \eta EH^T
$$
直观上，EH是所有词的output vector 的和乘以了权重e_j。可以认为上式，加上词典中每个output vector的一部分到input vector。如果yj > tj， wI的input vector倾向于远离wj的output vector。相反的如果yj < tj，wI的input vector倾向于接近wj的output vector。

在不断的更新过程中，词w的output vector会被共同出现的w的临近词 input vector反复的拖动，同样input vector也会被目标词的output vector反复拖动。直到趋于稳定。

## Optimizing Computational Efficiency

### Hierarchical Softmax

里面有V-1个inner-unit，所以空间复杂度并没变化。由于这种结构，对于词就没有了output vector representation。出于效率考虑，这个二叉树一般选择Huffman Tree。

### Negative Sampling

本文没有对负采样中的参数进行详细说明。

