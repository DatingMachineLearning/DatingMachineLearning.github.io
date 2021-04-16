# ID3, C4.5 和 CART 算法

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210328170928.svg" alt="决策树" style="zoom: 67%;" />

## ID3 算法

信息熵
$$
\mathrm{H}(X) = \sum_i^np_ilog_2(1/p_i) = -\sum_i^np_ilog_2(p_i)
$$
条件熵
$$
\begin{aligned}

H(Y \mid X) 
&=\sum_{i=1}^{n} p\left(x_{i}\right) H\left(Y \mid X=x_{i}\right) \\
&=-\sum_{i=1}^{n} p\left(x_{i}\right) \sum_{j=1}^{m} p\left(y_{j} \mid x_{i}\right) \log _{2} p\left(y_{j} \mid x_{i}\right) \\
&=-\sum_{i=1}^{n} \sum_{j=1}^{m} p\left(x_{i}, y_{j}\right) \log _{2} p\left(y_{j} \mid x_{i}\right)

\end{aligned}
$$
信息增益
$$
G(X) = H(Y) - H(Y|X)
$$


$Y = 1$ 表示买了。$Y = 0$ 表示没买。

$X = 1$ 表示附近学校好。$X = 0$ 表示附近学校不好。

觉得附近学校好，其中买的人有 5 个，不买的个数为 6 个；

觉得附近学校不好的，其中买的人有 1 个，不买的个数为 8 个；

可以得到概率：
$$
P(Y = 1|X = 1) = \frac{5}{11}\\
P(Y = 0|X = 1) = \frac{6}{11} \\
P(Y = 0 | X = 0) = \frac{8}{9} \\
P(Y = 1 | X = 1) = \frac{1}{9} \\
$$

各个条件熵：
$$
\begin{aligned}
H(Y=1|X = 1)  &= -\frac{5}{11}log_2(\frac{5}{11}) -\frac{6}{11}log_2(\frac{6}{11}) = 0.99 
\\
H(Y=1|X=0) &=  -\frac{1}{9}log_2(\frac{1}{9}) -\frac{8}{9}log_2(\frac{8}{9}) = 0.5\\
\end{aligned}
$$
按期望平均得到条件熵，算出 Y = 1 的信息熵：
$$
\begin{aligned}

&P(X = 1) =   \frac{11}{20}\\
&P(X =0) = \frac{9}{20}
\\ \\
&H(Y=1|X) =   \frac{11}{20} \times  0.99  +  \frac{9}{20} \times 0.5 = 0.77
\end{aligned}
$$

算出 Y = 1 的信息熵
$$
H(Y = 1) = -\frac{6}{20}log_2(\frac{6}{20}) -\frac{14}{20}log_2(\frac{14}{20}) = 0.88
$$
然后得出 X 事件的信息增益：
$$
G(X) = H(Y = 1) - H(Y = 1|X) = 0.88-0.77 = 0.11
$$

### 利用信息增益构建决策树

(案例出自西瓜书)

拿西瓜来说，他的样本属性可能是 $[色泽，瓜蒂，敲声，纹理,\dots]$，例如西瓜样本 

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411161337.png" alt="{3C0EB52A-E0E3-4D52-9B78-D62220062A5C}" style="zoom: 67%;" />

我们算出来所有属性的信息增益，D 是样本集合（如上图）：
$$
G(D，瓜蒂) = 0.143 \\
G(D，纹理) = 0.381 \\
G(D，脐部) = 0.289 \\
G(D，触感) = 0.006 \\
G(D，敲声) = 0.141
$$
此时，触感的信息增益最大，我们按照触感划分样本集合，得 D1, D2,  D3

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411162149.png" style="zoom:50%;" />
$$
G(D_1 ， 色泽) = 0.043\\ G(D_1 ，根蒂) = 0.458 \\ G(D_1 ，敲声) = 0.331 \\ G(D_1 ，脐部) = 0.458\\ G(D_1 ，触感) = 0.458
$$
……按照这种划分，我们就建立起了一棵决策树：

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210411162915.png" alt="{02BA7EDD-3E25-481F-82EE-52CBA23D1367}" style="zoom: 67%;" />

ID3 算法缺点：

1. 连续特征无法在ID3运用。
2. ID3 采用信息增益大的特征优先建立决策树的节点，在相同条件下，取值比较多的特征比取值少的特征信息增益大，这对预测性能影响很大。
3. ID3算法对于缺失值的情况没有做考虑。
4. 没有考虑过拟合的问题。

后面我们根据这三个问题逐一解决。

## C4.5 算法

### 信息增益比

信息增益准则对取值数目较多的属性有所偏好，ID3 算法的作者 Quinlan 基于上述不足，对ID3算法做了改进，不直接使用信息增益，而使用信息增益比：
$$
R_G(D, A) = \frac{G(D, A)}{IV_A(D)}
$$
D 是样本集合，A 是样本的某个属性，分母是样本 D 关于的属性 A 的固有值 (Intrinsic Value)：
$$
IV_D(A) = -\sum^n_i \frac{|D_i|}{|D|} log_2\frac{|D_i|}{|D|}
$$
属性 A 的某个取值越多，IV 的值就越大：
$$
IV_D(触感) = 0.874 (V = 2) \\ IV_D(色泽) = 1.580 (V = 3) \\ IV_D(编号) = 4.088 (V = 17)
$$
### 连续特征离散化

假设属性 A 的所有取值有 m 个，从小到大排列为 $a_1,a_2,...,a_m$ ，则 C4.5 取相邻两样本值的平均数，一共取得 $m-1$ 个划分点。对于这 $m−1$ 个点，分别计算以该点作为二元分类点时的信息增益。选择信息增益最大的点作为该连续特征的二元离散分类点。

比如取到的增益最大的点为 $a_t$ ,则小于 $a_t$ 的值为 $T_0$ 类别，大于 $a_t$ 的值为 $T_1$ 类别，这样我们就做到了连续特征的离散化。要注意的是，与离散属性不同的是，如果当前节点为连续属性，则该属性后面还可以参与子节点的产生选择过程。

具体来说，假设西瓜数据集有一个颜色深度属性，是被放缩到 [0, 1] 之间的连续值。

| 坏瓜 | 坏瓜 | 好瓜 | 好瓜 | 好瓜 | 坏瓜 | 好瓜 |
| ---- | ---- | ---- | :--- | ---- | ---- | ---- |
| 0.56 | 0.59 | 0.66 | 0.68 | 0.71 | 0.81 | 0.9  |

现在有 7 个数据。先计算相邻两样本值的平均数：

| at1   | at2   | at3  | at4   | at5  | at6   |
| ----- | ----- | ---- | ----- | ---- | ----- |
| 0.575 | 0.625 | 0.67 | 0.695 | 0.72 | 0.855 |

要在这些二元离散分类点找到增益最大的。为啥这里不需要信息增益比？因为所有二元分类点的属性都只有 $T_0$ 和 $T_1$ 。以 $a_{t3}$ 为例子，大于它的值好瓜有 3 个，坏瓜 1 个，小于它的值好瓜有 1 个，坏瓜有 2 个。下式假设 Y 事件为好瓜，$a_{t3}$ 事件为该点为分类点。
$$
\begin{aligned}
H(Y) &= -\frac{4}{7}log_2(\frac{4}{7})-\frac{3}{7}log_2(\frac{3}{7}) = 0.98\\
H(Y | T_1) &= -\frac{1}{4}log_2(\frac{1}{4})-\frac{3}{4}log_2(\frac{3}{4}) = 0.81 \\
H(Y | T_0) &= -\frac{1}{3}log_2(\frac{1}{3})-\frac{2}{3}log_2(\frac{2}{3}) =  0.91 \\
\\
G(a_{t3}) &= H(Y)  - [\frac{4}{7} H(Y | T_0) + \frac{3}{7} H(Y | T_1)] \\
&= 0.98 - [\frac{4}{7} H(Y | T_0) + \frac{3}{7} H(Y | T_1)]  \\
&= 0.12 \\
\end{aligned}
$$
找到最大增益值就可以确定分类点 $a_t$，根据它确定分支节点。

### 缺失值处理

为解决缺失值问题。



1. 在样本某些特征缺失的情况下选择划分的属性
2. 选定了划分属性，对于在该属性上缺失特征的样本的处理

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def main():
    iris = load_iris()
    print(iris["feature_names"])

    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X, y)


if __name__ == '__main__':
    main()
```

### 剪枝增强泛化

为了

## CART

### 基尼系数 



### 本文资料参考

[决策树（Decision Tree）-ID3、C4.5、CART比较](https://www.cnblogs.com/huangyc/p/9768858.html)

《机器学习》周志华





开放度 认知能力

