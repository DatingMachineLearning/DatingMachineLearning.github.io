考虑将：

> If you don't walk out, you will think that this is the whole world.

翻译到：

> 如果你不出去走走，你会以为这就是全世界。

## RNN 与 self-attention

对于翻译问题，单词序列 $X = \{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_n\}$ 翻译至单词序列 $Y = \{\mathbf y_1, \mathbf y_2, \cdots,  \mathbf y_m\}$，只考虑 Encoder 和 Decoder 结构，这里的实现假设只考虑 RNN。$\mathbf x_i,\mathbf y_i$ 都是单词的 embedding 向量，分别有 $n,m$ 个，其维度为 $d$。

上述序列分别是

 `[If, you, dont, walk, out, you, will, think, that, this, is, the, whole, world]`

`[如果，你，不，出去，走走，你，会，以为，这，就是，全，世界。]`

在这里 $n = 14$，$m = 12$ 。

对于时刻 $i$，Encoder 输出隐状态 $h_i$ ，一共有 n 个状态，
$$
\mathbf h_i = f_{\rm encoder}(\mathbf x_1,\mathbf x_2, \cdots, \mathbf x_i)\\
$$
每个 $\mathbf h_i$ 代表 $\mathbf x_i$ 及其之前单词的信息，或者说*中间语义表示*。

我们要利用它来生成，任意位置的单词翻译：
$$
\mathbf{y_i} = f_{\rm decoder}(\mathbf{h_i}, \mathbf{y_1}, \mathbf{y_2},\cdots,  \mathbf{y_{i-1}})
$$
然而 $\mathbf h_i$ 的长度是固定的，意味着无论我们的词个数 $n$ 有多大（无论多长的句子）都会被压缩到有限的状态向量中。这样做会损失信息。

为了解决这个问题，可以把所有不同时间的 $\mathbf h_i$ 堆叠成一个矩阵：
$$
H= \begin{bmatrix}
   \mathbf h_1^\top \\
   \mathbf h_2^\top \\
   \cdots \\
   \mathbf h_n^\top
\end{bmatrix}
$$
我们需要多少单词的信息，就可以生成多少大小的矩阵，于是将编码器改造成能输出所有状态 $\mathbf  h_i$ 。

### LSTM 编码器

LSTM 的编码器输出的隐状态 $\mathbf h_i$ ，能表达 $\mathbf x_i$ 之前很多单词的信息。biLSTM 则可以用 $\mathbf h_i $ （也就是任意位置）输出当前单词 $\mathbf x_i$ 及整个句子的语义表示。

## 改造解码器

改造了编码器，还需要改造解码器，让解码器能接受所有的中间语义表示 $\mathbf h_i$。

在翻译问题中，我们并不会需要所有的中间语义表示，

> If you don't walk out, you will think that this is the whole world.
>
> 如果你不出去走走，你会以为这就是全世界。

像这里的 `whole world` 对应 `全世界` ，他的翻译不依赖前面的分句，可以直接翻译出来。需要用到对齐 ( align ) 技术，像我们翻译时也会让 `if` 翻译为 `如果`，将 `you` 翻译为 `你`，翻译有这种对应关系。

问题现在是如何从 $H$ 中找到我们需要的与 $\mathbf y_i$ 对应的语义表示 $h_s$。

一个直观的想法就是乘以一个 one-hot 向量，直接提取出对应的 $h_s$。例如下面提取 $h_1$ 的过程：

$$
H \cdot I_i= 
\begin{bmatrix}
   1 \\
   0 \\
   \cdots \\
   0
\end{bmatrix} ^\top
\cdot
\begin{bmatrix}
   \mathbf h_1^\top \\
   \mathbf h_2^\top \\
   \cdots \\
   \mathbf h_n^\top
\end{bmatrix}

= h_1
$$
考虑到一个翻译应该和它源词汇附近的中间语义表示有关系，例如，对于翻译 `走走`来说， `walk` 和 `out` 的重要性是 0.7 和 0.2，而 `if` 和 `you` 的重要性是 0.03 和 0.05。

通过加权求和，就得到了我们的上下文向量 $c$
$$
c = H \cdot I_i= 
\begin{bmatrix}
   0.03 \\
   0.05 \\
   \cdots \\
   0.7 \\
   0.2 \\
   \cdots \\
   0.01
\end{bmatrix} ^\top

\cdot

\begin{bmatrix}
   \mathbf h_1^\top \\
   \mathbf h_2^\top \\
   \cdots \\
   \mathbf h_4^\top \\
   \mathbf h_5^\top \\
   \cdots \\
   \mathbf h_{14}^\top
\end{bmatrix} 

= \mathbf h_s
$$
加权和计算代替了“选择”向量的操作。那么如何才能得到这样的权重呢？

**attention 就是一种不需要人工也可以找到 if 和 `如果` 对应权重的机制。**

我们的目标是算出当前的 $\mathbf h_s$ 跟其他所有的中间语义表示 $\mathbf h_i$ 的相似度。越相似，越能代表当前 $\mathbf h_i$ 的重要性。
$$
\alpha_i = {\rm align}(\mathbf h_i, \mathbf h_n)
$$
一种方法就是内积，还有就是余弦相似度。

优点：

- 更多地关注了与当前输入相关的上下文。

## 没有 RNN 的 attention

将上述的编码器和解码器的结构抽象化，构建一个 attention 层。

假设我们要学习序列 $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_n\}$ 到 $\{\mathbf y_1, \mathbf y_2, \cdots,  \mathbf y_m\}$ 的关系，前者是编码器的输入，后者是解码器的输入。

基于 $\{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_n\}$ ，得出 key 和 value。通过可学习的参数 $W_k$ 和 $W_v$ ，分别得到 $\mathbf k$ 和 $\mathbf v$ 向量：
$$
\mathbf k_i = W_k\mathbf x_i \\
\mathbf v_i = W_v\mathbf x_i \\
$$
基于$\{\mathbf y_1, \mathbf y_2, \cdots,  \mathbf y_m\}$，得到 query，同样借由 $W_q$ 得到向量 $\mathbf q$：
$$
\mathbf q_j = W_q\mathbf y_j
$$

分别得出 $K$, $V$, $Q$ 矩阵，维度分别为 $n\times d, n\times d,m \times d $（在这里，列数是词嵌入的维度大小，如果是多头注意力，列数一般是 $d / h$， $h$ 为头数量） ：

$$
K =
\begin{bmatrix}
   \mathbf k_1^\top \\
   \mathbf k_2^\top \\
   \cdots \\
   \mathbf k_{n}^\top
\end{bmatrix},
V =
\begin{bmatrix}
   \mathbf v_1^\top \\
   \mathbf v_2^\top \\
   \cdots \\
   \mathbf v_{n}^\top
\end{bmatrix},
Q =
\begin{bmatrix}
   \mathbf q_1^\top \\
   \mathbf q_2^\top \\
   \cdots \\
   \mathbf q_{m}^\top
\end{bmatrix},
$$

来算权重 ，所有英语单词 $\{\mathbf x_i | 1 \le i \le n\}$ ( `[If, you, don't, walk, out, you, will, think, that, this, is, the, whole, world]`) 对某一个汉语词汇 $\mathbf y_j$ （`你`） 的权重向量。权重向量有 $m$ 个，每个的维度是 $n$：
$$
\mathbf{ \alpha}_j = {\rm softmax}(K \mathbf q_i)
$$

相当于所有的英语单词的 key ，分别和当前的 `如果` 的 query 做内积，越相似的权值越大。再对所有英语单词的 value 对应相乘，得到上下文向量：
$$
\begin{aligned}
\mathbf c_j &= \alpha_{j}^1\mathbf v_{1}  + \alpha_{j}^2\mathbf v_{2}  +  \cdots + \alpha_{j}^n  \mathbf v_{n} 
\\
\iff
\mathbf c_j &=  V^\top \cdot \mathbf{ \alpha}_j
\end{aligned}
$$
可以得到 $m$ 个上下文向量，维度为 $n$ 。

化简为矩阵计算，要得到所有的汉语词汇权重，可以构成一个 $n \times n$ 的矩阵：
$$
\alpha =  {\rm softmax}( Q K^\top )
$$
进一步得到上下文矩阵，维度为 $n\times d$：
$$
C = {\rm softmax}( Q K^\top ) V
$$
一般来说要做一个正则化的缩放，让梯度更新更加稳定，于是得到我们的 attention 函数：
$$
C = {\rm Attention(Q,K, V)} = {\rm softmax}( \frac{Q K^\top}{\sqrt{d}} ) V
$$

更一般地，attention layer 就是：
$$
C = {\rm AttentionLayer}(X, Y)
$$
自注意力层 self-attention layer：
$$
C = {\rm AttentionLayer}(X, X)
$$






[transformer中为什么使用不同的K 和 Q， 为什么不能使用同一个值？ - 知乎](https://www.zhihu.com/question/319339652)

[深度学习attention机制中的Q,K,V分别是从哪来的？ - 知乎](https://www.zhihu.com/question/325839123)

[transfomer里面self-attention的Q, K, V的含义 - 知乎](https://zhuanlan.zhihu.com/p/158952064)

[深度学习中的注意力模型（2017版） - 知乎](https://zhuanlan.zhihu.com/p/37601161)

[如何理解attention中的Q,K,V？ - 知乎](https://www.zhihu.com/question/298810062/answer/513421265)

[transformer中为什么使用不同的K 和 Q， 为什么不能使用同一个值？ - 知乎](https://www.zhihu.com/question/319339652)

