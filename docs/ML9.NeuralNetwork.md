# 神经网络

![visitor badge](https://visitor-badge.glitch.me/badge?page_id=xrandx.Dating-with-Machine-Learning)

## 1 直觉理解

### 1.1 感知机的问题

感知机可以学习 AND 函数，也可以学习 OR 函数。

AND 函数（与运算，全真则为真，否则为假）可以表示成数据：
$$
\mathbf{X} = 
\left [
\begin{aligned}
0&&0 \\
0&&1 \\
1&&0 \\
1&&1\\
\end{aligned} 
\right ]
\\
\mathbf y = \left [
\begin{aligned}
-1\\
-1\\
-1\\
1
\end{aligned} 
\right ]
$$
![image-20210525094701014](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210525094701.png)

由图知我们可以用一条线把正负类分开。（红色为正类，蓝色为负类，下面相同）

OR 运算（或运算，有真则为真，否则为假）可以表示成数据：
$$
\mathbf{X} = 
\left [
\begin{aligned}
0&&0 \\
0&&1 \\
1&&0 \\
1&&1\\
\end{aligned} 
\right ]
\\
\mathbf y = \left [
\begin{aligned}
-1\\
1\\
1\\
1
\end{aligned} 
\right ]
$$
![image-20210525093927687](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210525093927.png)

同样，我们可以用一条线把正负类分开。

**传统感知机只能处理线性可分的数据集。**

当我们遇到想模拟 XOR （异或运算，不同为真，相同为假）数据集时，数据如下
$$
\mathbf{X} = 
\left [
\begin{aligned}
0&&0 \\
0&&1 \\
1&&0 \\
1&&1\\
\end{aligned} 
\right ]
\\
\mathbf y = \left [
\begin{aligned}
-1\\
1\\
1\\
-1
\end{aligned} 
\right ]
$$
![image-20210525094628008](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210525094628.png)

由图可知，这个数据集不是线性可分的，因此不是感知机可以处理的。或者说，这是一个非线性的问题，通过线性集成的方法解决不了。

### 1.2 复合函数 -> 万能函数逼近器

如果可以用感知机模拟 OR 和 AND 运算（感知机当然也可以模拟 NOT 运算，遇真变假，遇假变真，相当于感知机在 x 轴上将 0 分为正类，1 分为负类）。那能不能把他们组合起来，就像计算机组成那样，组合成 XOR 运算？——这样的想法是自然的。我们知道
$$
XOR(x_1,x_2) = OR(AND(-x_1, x_2), AND(x_1, -x_2))
$$
或者熟悉的数学形式是：
$$
x_1 \oplus x_2 = (\neg x_1 \and x_2)\or (x_1\and \neg x_2)
$$
他的计算图是：

![img](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210525104713.png)

（一般来说，感知机的输出层（最后一层神经元）会有一个激活函数，也许是 Sigmoid 函数）

感知机层数在增加，模型的复杂度也在增加，使最后得到的 Output 能解决一些非线性的复杂问题。这样我们就构造了一个多层的感知机，可以模拟 XOR 运算。我们介绍的这种多层感知机（multi­-layer perceptron）模型其实就是 Neural Network。

准确来说，**多层感知机是神经网络的一个子集。**

对分类问题的拟合：

![1 e49COmHaPpL0TmaYG6r4sw](D:\file_backups\Dating Machine Learning\docs\1 e49COmHaPpL0TmaYG6r4sw.png)

对回归问题的拟合：

![5-Figure3-1](D:\file_backups\Dating Machine Learning\docs\5-Figure3-1.png)


### 1.3 前向传播

![{ADB79685-DF60-4302-97EE-D9438F89076A}](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210525220526.png)

这是一个两层神经网络，输入层不算在内。这样的网络可以看作是输入的 $x_0, x_1,x_2$ 经过从左到右的运算：

第 0 层：**输入层**，$a^{[0]} = [x_0,x_1,x_2]^\top$

第 1 层：**隐藏层**，$z^{[1]} = {\bf W}^{[1] \top} a^{[0]}, a^{[1]} = g(z^{[1]})$，先将上层输入线性组合，再通过激活函数得到了这一层。

第 3 层：**输出层**，$z^{[2]} = {\bf W}^{[2] \top} a^{[1]},a^{[2]} = g( z^{[2]}) $，先将上层输入线性组合，再通过激活函数得到了这一层。

实际上，神经网络是一个复杂的复合函数：
$$
\hat{y} = f({\bf x})= g[{\bf W}^{[2] \top} g({\bf W}^{[1] \top} {\bf x})]
$$

### 1.4 激活函数和代价函数

#### 1.4.1 激活函数

如果我们把已知的线性模型 linear regression, logistic regression(LR) **增加嵌套层次**，同样也可以提高泛化能力。前者可以解决回归问题，后者可以解决分类问题。

先选择分类问题来谈论神经网络。

若每个节点的 transformation function 都是线性运算，那么由每个节点的线性模型组合成的神经网络模型也必然是线性的。这跟直接使用一个线性模型在效果上并没有什么差异，模型能力不强，反而花费了更多不必要的力气。所以一般来说，中间节点不会选择线性模型。

**没有激活函数或者只有线性的激活函数的神经网络本质上只是一个线性回归模型。**换句话说，我们需要非线性的激活函数来增加模型的拟合能力，增强非线性拟合的能力。

Sigmoid 函数是应用广泛的非线性激活函数之一，也就是之前的对数几率函数。我们可以用 Sigmoid 函数作为每层的非线性变换。
$$
g(x) = \frac{1}{1+e^{-x}}
$$


<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210315174317.png" alt="C886BFB9-0C77-4894-B1CF-37409315BB57" style="zoom: 33%;" />

利用 Sigmoid 的神经网络就相当于多层 logistic 回归模型的集成嵌套。下一步，我们还需要为神经网络找到合适的代价函数和最小化代价的方法。

#### 1.4.2 代价函数

一个自然的想法是，像 logistic 回归一样利用平方损失函数。遗憾的是，对于二分类问题来说，平方损失函数效果并不好。这里采用交叉熵函数作为损失函数：
$$
L(y, \hat y) = -yln(\hat{y})-(1-y)ln(1-\hat{y})
$$
其中 $y \in\{ 1, 0 \}$ 表示正类或者负类，$\hat y \in \R$ 是我们预测结果为正类的概率。于是：
$$
\hat y =g({\bf W^\top a^{[i]}})
$$
可以发现一个性质：

当 $y = 0$，也就是实际为负类，则 $$ H = -ln(1-\hat y)$$，等价于预测为负类的概率的负对数。

当 $y = 1$，也就是实际为正类，则 $$ H = -ln(\hat y)$$ ，等价于预测为正类的概率的负对数。

使其最小化，可以兼顾正负类，相当于使其预测概率最大化。

### 1.5 反向传播

#### 1.5.1 梯度下降与学习率

寻找最值一个常用的方法就是求导，但在实际应用根本不可能这么理想，一定找得到唯一解。这时候就必须要靠找近似解的方式去逼近极值。

梯度下降法 (gradient descent) 是最优化理论里面的一个一阶找最佳解的一种方法，主要是希望用梯度下降法找到函数(刚刚举例的式子)的局部最小值，因为梯度的方向是走向局部最大的方向，所以在梯度下降法中是往梯度的反方向走。 

![duhksd7n](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210602194702.gif)



![](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210602200022.gif)

梯度下降法并不是下降最快的方向，它只是目标函数在当前的点的切平面（超平面）上下降最快的方向。

#### 1.5.3 反向传播

如果全为0就不能算梯度了，随机初始化 w



### 1.6 常用的激活函数

[Activation Functions — All You Need To Know!](https://medium.com/analytics-vidhya/activation-functions-all-you-need-to-know-355a850d025e)

### 1.6.1 sigmoid

在什么情况下适合使用 Sigmoid激活函数呢？

- Sigmoid 函数的输出范围是 0 到 1。由于输出值限定在 0 到 1，因此它对每个神经元的输出进行了归一化；
- 用于将预测概率作为输出的模型。由于概率的取值范围是 0 到 1，因此 Sigmoid 函数非常合适；
- 梯度平滑，避免「跳跃」的输出值；
- 函数是可微的。这意味着可以找到任意两个点的 sigmoid 曲线的斜率；
- 明确的预测，即非常接近 1 或 0。

Sigmoid激活函数有哪些缺点？

- 倾向于梯度消失；
- 函数输出不是以 0 为中心的，这会降低权重更新的效率；
- Sigmoid 函数执行指数运算，计算机运行得较慢。

### 1.6.2 Tanh / 双曲正切激活函数

![img](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210603165607.png)

tanh 是一个双曲正切函数。tanh 函数和 sigmoid 函数的曲线相对相似。但是它比 sigmoid 函数更有一些优势。

- 首先，当输入较大或较小时，输出几乎是平滑的并且梯度较小，这不利于权重更新。二者的区别在于输出间隔，tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好；
- 在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零。

注意：在一般的二元分类问题中，tanh 函数用于隐藏层，而 sigmoid 函数用于输出层，但这并不是固定的，需要根据特定问题进行调整。

### 1.6.3  ReLU激活函数

$$
ReLU({\bf z}) = \max(\mathbf z, 0) 
$$

- 当输入为正时，不存在梯度饱和问题。
- 计算速度快得多。ReLU 函数中只存在线性关系，因此它的计算速度比 sigmoid 和 tanh 更快。

当然，它也有缺点：

1. Dead ReLU 问题。当输入为负时，ReLU 完全失效，在正向传播过程中，这不是问题。有些区域很敏感，有些则不敏感。但是在反向传播过程中，如果输入负数，则梯度将完全为零，sigmoid 函数和 tanh 函数也具有相同的问题；
2. 我们发现 ReLU 函数的输出为 0 或正数，这意味着 ReLU 函数不是以 0 为中心的函数。

leaky ReLU




## 2 算法步骤

输入：

数据集 $D=\{({\bf x_i, y_i})\}^m_{i=1}$，设定初始的层数 $n$，学习率 $\eta$，各层的权值 $W^{[i]}$ ，阈值函数 $g(\cdot)$。

过程：

$\text{for all }({\bf x_i, y_i}) \in D \text{ do}$ 

​	$\text{for i-th layer in all layers do}$

​		计算第 $i$ 层输出 $a^{[i]}$， $a^{[i]} = g(W^{[i]\top} a^{[i - 1]})$

​		计算第 $i$ 层的梯度 $\mathrm{d}W^{[i ]}$

​		更新第 $i$ 层的权值 $W^{[i]} := W^{[i]} - \eta\cdot \mathrm{d}W^{[i ]}  $

$\text{end for}$

$\text{until }$ 达到指定条件



## 3 应用场景

### 3.1 神经网络适用于什么问题？

### 3.2 如何选择激活函数？

[Activation Functions | Fundamentals Of Deep Learning (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/)

- Sigmoid 函数以及它们的联合通常在**分类器（二元分类问题）**中有更好的效果；
- 由于梯度爆炸的问题，在某些时候需要避免使用 Sigmoid 和 Tanh 激活函数；
- ReLU 函数是一种常见的激活函数，学习的速度很快，而且和 Leaky ReLU 不容易使得激活函数的导数为 0，在目前使用是最多的；记住，ReLU永远只在隐藏层中使用。根据经验，我们一般可以从 ReLU 激活函数开始，但是如果 ReLU 不能很好的解决问题，再去尝试其他的激活函数。
- 如果遇到了一些死的神经元，我们可以使用 Leaky ReLU 函数；



## 4 用 numpy 实现神经网络

 



## 5 推导和证明

### 5.1 交叉熵与似然函数

假设实例 $\bf x_i$ 的某一个属性满足伯努利分布（二项分布），也就是
$$
{\bf x}_i^j 
$$







### 5.2 反向传播

$$
E_{in}(\hat{\bf y}, {\bf y}) = L(g[{\bf W}^{[2] \top} g({\bf W}^{[1] \top} {\bf X})], {\bf y})
$$

先看单个 w 怎么求，对 w2 的偏导

$$
\begin{aligned}
\frac{\partial E_{in}}{\partial {w}^{[2]}}  
&= \frac{\partial L}{\partial g} \cdot\frac{\partial g}{\partial {z}^{[2]}} \frac{\partial {z}^{[2]}}{\partial { w}^{[1]}}  \\
&= \frac{\partial L}{\partial z^{[2]}} \cdot   \frac{\partial {z}^{[2]}}{\partial { w}^{[1]}}
\end{aligned}
$$

对 w1 的偏导

$$
\begin{aligned}
\frac{\partial E_{in}}{\partial { w}^{[1]}} 

&= \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial z^{[2]}} \frac{\partial {z}^{[2]}}{\partial { a}^{[1]}} 
\cdot 
\frac{\partial g}{\partial z^{[1]}} \frac{\partial {z}^{[1]}}{\partial{w}^{[1]}}
\\
&=  \frac{\partial L}{\partial z^{[1]}} \cdot   \frac{\partial {z}^{[1]}}{\partial{w}^{[1]}}
\end{aligned}
$$
假设有第 0 层
$$
\begin{aligned}
\frac{\partial E_{in}}{\partial { w}^{[0]}} 

&= \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial z^{[2]}} \frac{\partial {z}^{[2]}}{\partial { a}^{[1]}} 
\cdot 
\frac{\partial g}{\partial z^{[1]}} \frac{\partial {z}^{[1]}}{\partial{a}^{[0]}} 
\cdot  
\frac{\partial g}{\partial z^{[0]}} \frac{\partial {z}^{[0]}}{\partial{w}^{[0]}} \\

&= \frac{\partial L}{\partial z^{[0]}}  \cdot   \frac{\partial {z}^{[0]}}{\partial{w}^{[0]}}  \\

\end{aligned}
$$
也就是：
$$
\frac{\partial E_{in}}{\partial { w}^{[i]}}  = \frac{\partial L}{\partial z^{[i]}}  \cdot   \frac{\partial {z}^{[i]}}{\partial{w}^{[i]}}  = \frac{\partial L}{\partial z^{[i]}}  \cdot a^{[i -1]}\\
$$
对 z 有递推式：
$$
\begin{aligned}

\frac{\partial L}{\partial z^{[i]}} 
&= \frac{\partial L}{\partial a^{[i]}} \frac{\partial g}{\partial z^{[i]}} 
\end{aligned}
$$
关于 a 的偏导：
$$
\begin{aligned}

\frac{\partial L}{\partial a^{[i]}} 
&= \frac{\partial L}{\partial z^{[i + 1]}} \frac{\partial  z^{[i + 1]}}{\partial a^{[i]}} \\
&= \frac{\partial L}{\partial z^{[i + 1]}} w^{[i + 1]}

\end{aligned}
$$
对于矩阵而言
$$
\mathrm{d} Z^{[i]} =  \mathrm{d} A^{[i]} * \mathrm{d} g \\
\mathrm{d} W^{[i]} = \frac{1}{m} \mathrm{d} Z^{[i]} \cdot A^{[i]}\\
\mathrm{d} A^{[i-1]} =   W^{[i ]} *  \mathrm{d}Z^{[i]} \\
$$


---

## 多层神经网络





### 梯度下降法



1. 不可逆函数
2. 


一个标准的机器学习处理流程是

```mermaid
graph LR
	数据获取及分析-->数据预处理-->特征工程-->训练模型选择与调优-->后处理-->模型评估

```





1. 为什么要以平方误差为损失函数？

2. 样本归一化：预测时的样本数据同样也需要归一化，但使用训练样本的均值和极值计算，这是为什么？

3. 当部分参数的梯度计算为0（接近0）时，可能是什么情况？是否意味着完成训练？

4. 随机梯度下降的 batch-size设置成多少合适？过小有什么问题？过大有什么问题？提示：过大以整个样本集合为例，过小以单个样本为例来思考。

5. 一次训练使用的配置：5个epoch，1000个样本，batch-size = 20，最内层循环执行多少轮？


答案：

> 1. 平方误差函数有曲度，更容易寻找到
>
> 2. 因为在真实的世界我们是不可能知道要预测的样本的均值和极值的。
>
> 3. 不一定。也可能是因为偏导数为 0，
>
> 4. 128（batch size 需要调参）
>
>    过大：
>
>    - batch size太大，memory容易不够用。这个很显然，就不多说了。
>
>    - batch size太大，深度学习的优化（training loss降不下去）和泛化（generalization gap很大）都会出问题。
>
>    过小：
>
>    - 当你的 batch size太小的时候，在一定的epoch数里，训练出来的参数 posterior 是根本来不及接近 long-time limit 的定态分布。来不及收敛。
>
>    合适的batch size范围主要和收敛速度、随机梯度噪音有关。[怎么选取训练神经网络时的Batch size? - 知乎 (zhihu.com)](https://www.zhihu.com/question/61607442)
>
>    [训练神经网络时如何确定batch的大小？ (qq.com)](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484570&idx=1&sn=4c0b6b76a7f2518d77818535b677e087&chksm=970c2c4ca07ba55ad5cfe6b46f72dbef85a159574fb60b9932404e45747c95eed8c6c0f66d62#rd)
>
> 5. 最外层 5 次，最内层 50 次。