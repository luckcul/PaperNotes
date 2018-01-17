# BLEU: a Method for Automatic Evaluation of Machine Translation 

本文提出了BLEU(bilingual evaluation understudy)指标来衡量机器翻译的质量。提出该指标的背景是，对于机器翻译人工评价质量的代价非常高而且无法复用，因而提出了自动评估的BLEU指标。

## Introduction

机器翻译的人工评估要考虑多个方面，包括adequacy, fidelity, fluency等。

如何评价翻译的效果呢？The closer a machine translation is to a professional human translation, the better it is. 这也是自动化评估指标的目标，所以指标要可以**量化**机器翻译与其最接近的一个或多个人工翻译的距离。

## Modified n-gram precision

如果是简单的n-gram precision计算，可以这么做，`机器翻译语句中出现在参考翻译的n-gram的数量 / 机器翻译语句中所有n-gram的数量`。 然而这么计算会导致一些问题，例如：

```
candidate: the the the the the the the 
Reference 1: The cat is on the mat. Reference
Reference 2: There is a cat on the mat.
```

这样得到的准确度是7/7，然而candidate的质量却很差。

因此需要modified n-gram precision，首先，计算一个n-gram在所有参考翻译中出现的最大次数`Max_Ref_Count`；然后，用这个值进行修正，得到`Count_clip = min(Count, Max_Ref_count)`，其中Count为该n-gram在机器翻译中出现的次数；最后，得到修正的n-gram准确度 `sum(Count_clip) / total candidate n-gram`。这样求出来的准确度是2/7。

这样的准确度其实捕获了翻译的两个方面：adequacy和fluency。

## Ranking systems using only modified n-gram precision

利用上面的准确度算法，对于每一个句子都可以得到modified n-gram precision。然而一个句子不能代表文本翻译水平，于是可以把一段或者所有的句子综合起来，得到准确度`pn`。
$$
p_n = \frac{\sum_{C\in \{ Candidates\}}\sum_{n-gram \in C} Count_{clip}(n-gram)}{\sum_{C'\in \{ Candidates\}}\sum_{n-gram \in C'} Count(n-gram')}
$$

## Combining the modified n-gram precisions

通过实验可以发现，modified n-gram precision随着n的增大大致呈指数衰减，所以要采用一种平均策略考虑进去这种指数衰减。BLEU采用的是average logarithm with uniform weights，等价于使用geometric mean。实验发现，最大值N取4时与人工评估最好的相关性。
$$
p_{combining}=exp(\sum_{n=1}^{N}w_nlog\ p_n)
$$

## Sentence brevity penalty

机器翻译的结果不应该过长或者过短。如果比较短，中的n-gram几率会比较大，那么它的modified n-gram precision会比较高。而较长时，modified n-gram precision已经惩罚了。所以应该对短句子设置长度惩罚，BLEU引入了简短惩罚因子。如果逐句计算惩罚项，那么短句子会被惩罚的过于严格。因而在整个语料库上计算惩罚项比较合理。在测试语料库中，将每个候选语句在参考语句中的最佳匹配长度(和候选语句长度最接近的参考语句长度)累加在一起得到reference length r，再讲所有参考语句的长度累加得到 candidate length c。
$$
BP=
\begin{equation}  
\left\{  
             \begin{array}{lr}  
             1 & if\ c > r\\  
             e^{(1-r/c)} & if\ c \leq r   
             \end{array}  
\right.  
\end{equation}  
$$

## BLEU detail

结合上面的叙述，可以得到BLEU的公式。本文选取N=4，w_n=1/N。
$$
BLEU = BP\cdot exp(\sum_{n=1}^{N}w_n\ log\ p_n)
$$

## Comments

* 之前一直以为测试语料库的BLEU值是每个句子的BLEU值的平均，其实，是整个语料库上进行计算准确度的。
* BP不惩罚长度比较大的句子是因为在计算modified precision中已经蕴含了对长度较大的惩罚。
* 关于具体BLEU具体的实现，可以参考[nltk.translate.bleu_score](http://www.nltk.org/_modules/nltk/translate/bleu_score.html)。 