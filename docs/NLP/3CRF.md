# 条件随机场

> [NLP实战-中文命名实体识别](https://mp.weixin.qq.com/s/k9njcx_11ELsmfH1mGO28Q)
>
> [如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？ - 交流_QQ_2240410488 - 博客园](https://www.cnblogs.com/jfdwd/p/11158652.html)
>
> [CRF Layer on the Top of BiLSTM - 5 | CreateMoMo](https://createmomo.github.io/2017/11/11/CRF-Layer-on-the-Top-of-BiLSTM-5/)
>
> [luopeixiang/named_entity_recognition: 中文命名实体识别（包括多种模型：HMM，CRF，BiLSTM，BiLSTM+CRF的具体实现）](https://github.com/luopeixiang/named_entity_recognition)
>
> [手撕算法｜随机场 - 钱小z](http://www.qianxz.pro/2020/11/09/mrf-crf/)

HMM 模型中存在两个假设：

- 输出观察值之间严格独立
- 状态转移过程中当前状态只与前一状态有关。也就是说，在命名实体识别的场景下，HMM认为观测到的句子中的每个字都是相互独立的，而且当前时刻的标注只与前一时刻的标注相关。

但实际上，命名实体识别往往需要更多的特征，比如词性，词的上下文等等，同时当前时刻的标注应该与前一时刻以及后一时刻的标注都相关联。由于这两个假设的存在，显然HMM模型在解决命名实体识别的问题上是存在缺陷的。

而条件随机场就没有这种问题，它通过引入自定义的特征函数，不仅可以表达观测之间的依赖，还可表示**当前观测与前后多个状态之间的复杂依赖**，可以有效克服HMM模型面临的问题。

> 下文绝妙部分翻译自 [Introduction to Conditional Random Fields](https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)

想象一下，你有一系列关于你一个月生活的自拍照，你想用它代表的活动标记每个图像（吃，睡觉，驾驶等）。 你打算怎么做？

一种方法是忽略自拍的顺序性，并构建每个图像分类器。 例如，考虑到一个月的标记快照，你可以了解凌晨6点拍摄的黑暗图像往往是睡觉，有很多鲜艳色彩的图像往往是关于跳舞，汽车的图像是关于驾驶的，等等。

但是，通过忽略这个顺序方面，你丢失了很多信息。 

例如，如果你看到嘴的特写图片会发生什么 - 是关于唱歌或吃什么？ 如果你知道以前的图像是你吃或烹饪的照片，那么这幅画更有可能关于进食的; 然而，如果之前的图像包含你唱歌跳舞，那么这可能代表了唱歌。

因此，为了提高我们标签程序的准确性，我们应该包含附近照片的标签，这正是条件随机场所做的。

## 词性标记 Part-of-Speech Tagging

让我们更详细地进入一些常见的词性标记 (POS) 的常见例子。

在 POS 标记中，目标是用标签标记一个句子（一系列 word 或 token），例如（ADJECTIVE, NOUN, PREPOSITION, VERB, ADVERB, ARTICLE）等标签。

例如，鉴于“Bob drank coffee at Starbucks”的句子，标签可能是“Bob（名词）drank （动词）coffee （名词）at （介词）Starbucks（名词）”。

因此，让我们建立一个 CRF ，以用他们的言论标记句子。 就像任何分类器一样，我们首先需要决定一个 **特征函数**。

### CRF 的特征函数

在 CRF 中, 每个特征函数 $f$ 有以下输入：

- 句子 $s$
- 单词在句子中的位置 $i$
- 当前单词的标签 $l_i$​
- 前一个单词的标签 $l_{i-1}$

并输出一个实数（尽管数字通常只是0或1）。

(注意：通过限制我们的特性只依赖于当前和以前的标签，而不是整个句子中的任意标签，我实际上是在构建一个 **线性链CRF** 的特殊情况。为了简单起见，我将在这篇文章中忽略一般的 CRF )

例如，当前一个单词是“Very”，一个特征函数可以测量我们有多怀疑当前单词应该被标记为形容词。

### 从特征到概率

然后，通过给每一个特征函数 $f_j$ 一个权重 $\lambda_j$ （将在下面谈谈如何从数据中学习这些权重），给出一个句子 $s$​，我们现在可以通过给一个句子中的所有单词上添加权重特征计算标签 $l$ 的得分：
$$
score(l | s) = \sum_{j = 1}^m \sum_{i = 1}^n \lambda_j f_j(s, i, l_i, l_{i-1})
$$
第一个求和遍历每个特征函数 $j$，内部求和遍历句子的每个位置。

最后，我们可以将这些分数转化为概率 $p(l|s)$，通过指数和标准化在0和1之间：
$$
p(l | s) = \frac{exp[score(l|s)]}{\sum_{l’} exp[score(l’|s)]} = \frac{exp[\sum_{j = 1}^m \sum_{i = 1}^n \lambda_j f_j(s, i, l_i, l_{i-1})]}{\sum_{l’} exp[\sum_{j = 1}^m \sum_{i = 1}^n \lambda_j f_j(s, i, l’_i, l’_{i-1})]}
$$

## 特征函数的例子

那么这些特征函数是什么样的？ POS标记特征的例子有：
$$
f_1(s, i, l_i, l_{i-1}) = 
\left\{
\begin{aligned}
1,  &l_i = {\rm  ADVERB} \text{且以 -ly 结尾} \\
0, &\text{否则}
\end{aligned}
\right.
$$

如果与此特征相关联的权重 $λ_1$ 大于零而且很大，则此特征基本上是指我们更喜欢将以 "-ly" 结尾的单词标记为 ADVERB。

$$
f_2(s, i, l_i, l_{i-1}) = 
\left\{
\begin{aligned}
1, & \text{当 i = 1 ,}l_i  = {\rm VERB}\text{且句子 s 以问号结尾}\\
0, &\text{否则}
\end{aligned}
\right.
$$

同样，如果与此特征相关的权重 $λ_2$ 是大的且为正的，那么将 VERB 分配给问句中的第一个单词的标签 (例如，“Is this a sentence beginning with a verb?”) 是首选。

$$
f_3(s, i, l_i, l_{i-1}) = 
\left\{
\begin{aligned}
1, &\text{当 i = 1,} l_{i-1}  = {\rm ADJECTIVE} \text{且} l_{i}  ={\rm NOUN}  \\
0, &\text{否则}
\end{aligned}
\right.
$$

同样，这个特征的权重为正意味着形容词后面往往跟着名词。
$$
f_4(s, i, l_i, l_{i-1}) = 
\left\{
\begin{aligned}
1, &\text{当 i = 1, }  l_{i-1}  = {\rm PREPOSITION} \text{且}   l_{i}  = {\rm PREPOSITION }\\
0, &\text{否则}
\end{aligned}
\right.
$$
负权重 $\lambda_4$ 意味着介词后面一般不是介词，所以当这种情况发生的时候，我们应该尽量避免打上标签。

那就是它！ 总结：要构建 CRF，你只需定义一系列特征函数（这可以取决于整个句子，当前位置，和附近的标签），将其分配权重，并将它们全部添加在一起，必要时在最后转换为概率。

现在让我们退后一步，并将 CRF 与其他一些常用的机器学习技术进行比较。

## 有点像对数几率回归


$$
p(l | s) = \frac{exp[\sum_{j = 1}^m \sum_{i = 1}^n f_j(s, i, l_i, l_{i-1})]}{\sum_{l’} exp[\sum_{j = 1}^m \sum_{i = 1}^n f_j(s, i, l’_i, l’_{i-1})]}
$$
这是因为 CRF 实际上基本上是逻辑回归的序列（sequential）版本：对数几率回归是分类的对数线性模型，而 CRFs 是序列标签的对数线性模型。

对数几率的公式是：
$$
p(l|s ) = \frac{exp(w^\top f+b)}{1 + exp(w^\top f+b)}
$$


## 有点像 HMM

回想一下隐马尔可夫模型是词性标注(以及通常的序列标注)的另一种模型。CRF们将任何一堆函数放在一起来获得标签得分，而 hmm 则采用生成方法来进行标签和定义：
$$
p(l,s) = p(l_1) \prod_i p(l_i | l_{i-1}) p(w_i | l_i)
$$

- $p(l_i | l_{i-1})$​​ 是状态转移概率。（例如：介词后面是名词的概率）
- $p(w_i | l_i)$​ 是观测概率。 (例如：从状态“名词” 观测到 "dad" 的概率)

HMM 与 CRF 相比如何呢？CRF 更强大，它们可以模拟 HMM 所能模拟的一切，甚至更多。可以这样看：

注意 HMM 的对数概率是
$$
\log p(l,s) = \log p(l_0) + \sum_i \log p(l_i | l_{i-1}) + \sum_i \log p(w_i | l_i)
$$




> [BiLSTM-CRF模型代码分析及CRF回顾 - Joven Chu Blog](https://jovenchu.cn/2020/06/09/2020-06-09-BiLSTM-CRF/)
>
> [手撕 BiLSTM-CRF - 知乎](https://zhuanlan.zhihu.com/p/97676647)
>
> [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF — PyTorch Tutorials 1.9.0+cu102 documentation](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)



## 标准形式

我们有输入的词序列：$X = x_1^n = x_1\dots x_n$

以及我们想要得到的对应输出标签序列：$Y = y_1^n = y_1 \dots y_n$

在 HMM 中我们计算最大可能标签序列，依赖的是基于贝叶斯法则和 $P(X|Y)$ 来最大化 $P(Y|X)$ ：
$$
\begin{aligned}
\hat{Y} &=\underset{Y}{\operatorname{argmax}} p(Y \mid X) \\
&=\underset{Y}{\operatorname{argmax}} p(X \mid Y) p(Y) \\
&=\underset{Y}{\operatorname{argmax}} \prod_{i} p\left(x_{i} \mid y_{i}\right) \prod_{i} p\left(y_{i} \mid y_{i-1}\right)
\end{aligned}
$$
在 CRF 中，我们计算的却是后验概率 $P(Y|X)$ 
$$
\hat{Y} = \mathop{\arg\max}_{Y\in\cal Y}  P(Y|X)
$$

### CRF 用于命名实体识别

CRF在每次步骤中都不会计算每个标签的概率。 另外，在每个 time step 中，CRF 通过一组相关的特征计算对数线性函数，并且这些局部特征被聚合并归一化以产生整个序列的全局特征。

CRF 就像多项式 logistic 回归对单个 token 的作用的一个巨大版本


$$
p(Y \mid X)=\frac{\exp \left(\sum_{k=1}^{K} w_{k} F_{k}(X, Y)\right)}{\sum_{Y^{\prime} \in {\cal Y}} \exp \left(\sum_{k=1}^{K} w_{k} F_{k}\left(X, Y^{\prime}\right)\right)}
$$
常见的归一化是这样的：
$$
\begin{aligned}
p(Y \mid X) &=\frac{1}{Z(X)} \exp \left(\sum_{k=1}^{K} w_{k} F_{k}(X, Y)\right) \\
Z(X) &=\sum_{Y^{\prime} \in {\cal Y}} \exp \left(\sum_{k=1}^{K} w_{k} F_{k}\left(X, Y^{\prime}\right)\right)
\end{aligned}
$$
这些 $F_k(\cdot) $ 函数是所谓的全局特征，因为每一个都是输入序列 X 和输出序列 Y 的属性。对于 Y 中的每一个位置 i，我们可以将其分解为局部特征之和：
$$
F_{k}(X, Y)=\sum_{i=1}^{n} f_{k}\left(y_{i-1}, y_{i}, X, i\right)
$$
这个局部特征函数只考虑：

在 CRF 中, 每个特征函数 $f$ 有以下输入：

- 整个句子 $X$
- 单词在句子中的位置 $i$
- 当前单词的标签 $y_i$​
- 前一个单词的标签 $y_{i-1}$

这样的 CRF 叫做线性链 CRF。

相比之下，一般化的 CRF 允许特征使用任何位置的标签，并且有必要依赖于较远的标签，如 $y_{i-4}$，一般化的 CRF 需要更加复杂的推断，并且不太常用于语言处理。

<img src="\img\20210729165054.png" alt="基于特征的NER系统的典型特征" style="zoom:67%;" />

基于特征的NER系统的典型特征

<img src="\img\1627548796(1).png" alt="1627548796" style="zoom:67%;" />

上图是一些典型的 NER 特征：词性、单词的长度大小、是否在词典，最后一列是标注结果。

## 模型推导和训练

如何找到序列 $X$ 的对应最佳的标注序列 $\hat Y$
$$
\begin{aligned}
\hat{Y} &=\underset{Y \in {\cal Y}}{\operatorname{argmax}} P(Y \mid X) \\
&=\underset{Y \in {\cal Y}}{\operatorname{argmax}} \frac{1}{Z(X)} \exp \left(\sum_{k=1}^{K} w_{k} F_{k}(X, Y)\right) \\
&=\underset{Y \in {\cal Y}}{\operatorname{argmax}} \exp \left(\sum_{k=1}^{K} w_{k} \sum_{i=1}^{n} f_{k}\left(y_{i-1}, y_{i}, X, i\right)\right) \\
&=\underset{Y \in {\cal Y}}{\operatorname{argmax}} \sum_{k=1}^{K} w_{k} \sum_{i=1}^{n} f_{k}\left(y_{i-1}, y_{i}, X, i\right) \\
&=\underset{Y \in {\cal Y}}{\operatorname{argmax}} \sum_{i=1}^{n} \sum_{k=1}^{K} w_{k} f_{k}\left(y_{i-1}, y_{i}, X, i\right)
\end{aligned}
$$

- $Z(X)$ 是一个不变量
- $exp(\cdot)$ 不影响单调性



- [lihang-code/11.CRF.ipynb at master · fengdu78/lihang-code](https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC11%E7%AB%A0%20%E6%9D%A1%E4%BB%B6%E9%9A%8F%E6%9C%BA%E5%9C%BA/11.CRF.ipynb)
- [Conditional Random Fields - Stanford University (By Daphne Koller) - YouTube](https://www.youtube.com/watch?v=rc3YDj5GiVM&ab_channel=MachineLearningTV)



