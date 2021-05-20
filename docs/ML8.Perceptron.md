# 感知机

![visitor badge](https://visitor-badge.glitch.me/badge?page_id=xrandx.Dating-with-Machine-Learning)

目前以深度学习为主的算法，采取的是一种连接主义观点。

## 感知机 Perceptron

### 假设

感知机是一种二分类的监督学习算法，能够根据实例向量，输出其类别，+1代表正类，-1代表负类。感知机属于判别模型，它的目标是要将输入实例通过分离超平面将正负二类分离。

感知机的假设是：正负二类数据都能被一个超平面分割。例如下面的两张图。

二维情况下数据可被一条线分离：

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422164637.png" alt="linearly-separable" style="zoom:50%;" />

三维情况下数据被一张平面分离：

![No0tXOKWTx39iV2CJLjCGaFGr1520492096_kc](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422163634.png)

怎么去判断一类是正类还是负类？

以第一张图为例，圆圈代表正类，叉代表负类。假设这条直线方程是 $x_1 + x_2 - 2 = 0$ （或者我们熟悉的 $y = -x + 2$），令 $h(\boldsymbol x) = x_1+x_2-2$。

代入正类数据 $\boldsymbol{x_1} = (2, 2)$，则 $h(\boldsymbol x) = 2 + 2 -2 = 2$ 。

代入负类数据 $\boldsymbol{x_2} = (0, 1)$，则 $h(\boldsymbol x) = 0 + 1 -2 = -1$ 。

根据高中知识，所有的在其上方的正类数据都会大于 0， 负类数据小于 0。

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422164637.png" alt="linearly-separable" style="zoom:67%;" />

这时我们可以利用数学中的符号函数：
$$
\mathrm{sgn}(x) = 
 \left\{
\begin{aligned}
1 &&\text{if } x>0 \\
0 &&\text{if } x=0 \\
-1 &&\text{if } x<0
\end{aligned}
\right.
$$
图像是

![Signum_function.svg](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422195725.png)

利用它我们就可以让直线或者超平面区分出不同类别，令偏置项 b 等于 w0：
$$
\begin{aligned}
g(x_1, x_2) &= \mathrm{sgn} ( w_1x_1 + w_2x_2 + b) \\
&=\mathrm{sgn} ( w_1x_1 + w_2x_2 + w_0 \times 1)
\end{aligned}
$$
简化：
$$
g(\mathbf x ) = \mathrm{sgn}(\mathbf{w^\top x})
$$

### 找到超平面

既然已经知道了如何判别正负类，下一个问题是怎么找到一个超平面，能把所有数据一分为二？

若 $y_i \cdot \mathrm{sgn}(\mathbf{w^\top x_i}) > 0$，说明二者同号，分类正确。若 $y_i \cdot \mathrm{sgn}(\mathbf{w^\top x_i}) < 0$，说明实际的 $y_i$ 和预测的结果不同，需要改变当前这个超平面。

当 $y_i \cdot \mathrm{sgn}(\mathbf{w^\top x_i}) < 0$， 有两种情况：

一是 $y_i = +1$，则 $\mathrm{sgn}(\mathbf{w^\top x_i}) = -1$， 说明  $\mathbf{w^\top x_i} < 0$ ，则 $\mathbf w$ 与 $\mathbf{x_i}$ 呈钝角。$\mathbf w$ 与当前的超平面垂直，如果我们让它与 $\mathbf{x_i}$ 夹角变为锐角，就能把 $\mathbf{x_i}$ 分类正确。如图

![image-20210423165028429](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210423165028.png)

二是 $y_i = -1$，则 $\mathrm{sgn}(\mathbf{w^\top x_i}) = +1$， 说明  $\mathbf{w^\top x_i} > 0$ ，则 $\mathbf w$ 与 $\mathbf{x_i}$ 呈锐角。同样，$\mathbf w$ 与当前的超平面垂直，如果我们让它与 $\mathbf{x_i}$ 夹角变为钝角，就能把 $\mathbf{x_i}$ 分类正确。如图

![image-20210423165039476](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210423165039.png)


注意，若 $\theta$ 为 $\mathbf w$ 和 $\mathbf {x_i}$夹角则：
$$
\mathrm{cos}(\theta) = \frac{\mathbf{w^\top x_i}} {||\mathbf w||\cdot ||\mathbf {x_i}||}
$$

综上所述，我们只需要
$$
\text{if }  y_i \cdot \mathrm{sgn}(\mathbf{w^\top x_i}) < 0
\\
\mathbf{w_{i+1}} \leftarrow \mathbf{w_i} + y_i \cdot \mathbf{x_i}
$$

### 线性可分

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210423180603.png" alt="hl_classif_separation" style="zoom:101%;" />

对于二维（两个特征）的数据集，如果存一条直线，能够把这两个分类完美区分，那么这个数据集就是线性可分的。

对于高维（很多特征）的数据集，如果存在一个超平面，能够把这两个分类完美区分，那么这个数据集也是线性可分的。否则，叫做线性不可分，线性不可分指有部分样本用线性分类面划分时会产生分类误差的情况。。

我们可以直观看出来，感知机算法只能处理线性可分的数据集才能正确。

### 学习 AND 函数

假设我们现在有数据集 X 和标签集 y，这样的数据很显然是一个逻辑「与」运算。（ 1 代表「是」，-1 代表「非」）
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
感知机算法如何学习这样的结果？

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    clf = Perceptron()
    clf.fit(X, y)
    print(clf.coef_)
    coef = clf.coef_[0]
    x = np.arange(-1, 2, 0.01)
    y = - (x * coef[0] + clf.intercept_) / coef [1]
    plt.plot(x, y)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
```



## 多层神经网络





### 梯度下降法



1. 不可逆函数
2. 







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

