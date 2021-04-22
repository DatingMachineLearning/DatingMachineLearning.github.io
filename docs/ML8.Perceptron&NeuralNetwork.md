# 感知机与神经网络

目前以深度学习为主的算法，采取的是一种连接主义观点。

## 感知机

感知机是一种二分类的监督学习算法，能够根据实例向量，输出其类别，+1代表正类，-1代表负类。感知机属于判别模型，它的目标是要将输入实例通过分离超平面将正负二类分离。

感知机的假设是：正负二类数据都能被一个超平面分割。例如下面的两张图。

二维情况下数据可被一条线分离：

<img src="https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422164637.png" alt="linearly-separable" style="zoom:50%;" />

三维情况下数据被一张平面分离：

![No0tXOKWTx39iV2CJLjCGaFGr1520492096_kc](https://gitee.com/xrandx/blog-figurebed/raw/master/img/20210422163634.png)

问题是我们的算法要怎么才能找到这样一个超平面？

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
g(\boldsymbol x ) = \mathrm{sgn}(\boldsymbol{w^\top x})
$$


## PLA 算法



## 多层神经网络