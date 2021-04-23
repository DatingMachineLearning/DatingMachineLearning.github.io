# 感知机与神经网络

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