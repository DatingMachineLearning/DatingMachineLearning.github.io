![visitor badge](https://visitor-badge.glitch.me/badge?page_id=xrandx.Dating-with-Machine-Learning)

## A

### A/B testing

A/B 测试，一种统计方法，用于将两种或多种技术进行比较，通常是将当前采用的技术与新技术进行比较。A/B 测试不仅旨在确定哪种技术的效果更好，而且还有助于了解相应差异是否具有显著的统计意义。A/B 测试通常是采用一种衡量方式对两种技术进行比较，但也适用于任意有限数量的技术和衡量方式。

### activation function

激活函数，一种函数（例如 ReLU 或S 型函数），用于对上一层的所有输入求加权和，然后生成一个输出值（通常为非线性值），并将其传递给下一层。

### AdaGrad

一种复杂的梯度下降算法，重新调节每个参数的梯度，高效地给每个参数一个单独的学习率。详见论文：http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf。

### AUC

曲线下面积，一种考虑到所有可能的分类阈值的评估标准。ROC 曲线下面积代表分类器随机预测真正类（Ture Positives）要比假正类（False Positives）概率大的确信度。

## B

### batch

批量，一个批量中样本的数量。例如，SGD 的批量大小为 1，而 mini-batch 的批量大小通常在 10-1000 之间。批量大小通常在训练与推理的过程中确定，然而 TensorFlow 不允许动态批量大小。



### benchmark 

benchmark 过程包括三个步骤：

1. 设置(setup): 根据实验目的做得设置，通常也是在论文实验结果之前要交代的实验设置，根据所要研究的问题选择合适的数据集、算法、对比算法、比较参数等等。
2. 执行(execution): 这个部分就是按照上一步的设置进行实验。
3. 分析(analysis): 通过各种分析方法分析上一步得到的实验结果，用来佐证提出的算法或者假设。

baseline 的目的是比较提出算法的性能或者用以比较彰显提出算法的优势。比如，[difference between baseline and benchmark in performance of an application](https://link.zhihu.com/?target=http%3A//stackoverflow.com/a/347029/222670) 里面举的一个微软测试操作系统启动时间的例子。在 Win7 的测试阶段，为了比较不同版本 Win7 的启动时间，在相同配置的电脑上安装XP-RTM, XP-SP2, Vista-RTM, Vista-SP1, Vista-SP2和之前的各个 beta 版本，这些系统就都是 baseline，因为对这些系统的启动过程，微软的工程师都很清楚。然后在相同配置的电脑上启动 Win7就能比较出来，在哪个环节出现了问题，导致了什么问题。

### binary classification

二元分类，一类分类任务，输出两个互斥（不相交）类别中的一个。例如，一个评估邮件信息并输出「垃圾邮件」或「非垃圾邮件」的机器学习模型就是一个二元分类器。

## T

### trade-off 

Bias和Variance是针对Generalization（泛化、一般化）来说的。在机器学习中，我们用训练数据集学习一个模型，我们通常会定义一个损失函数（Loss Function），然后将这个Loss（或者叫error）的最小化过程，来提高模型的性能（performance）。然而我们学习一个模型的目的是为了解决实际的问题（即将训练出来的模型运用于预测集），单纯地将训练数据集的Loss最小化，并不能保证解决更一般的问题时模型仍然是最优的，甚至不能保证模型是可用的。这个训练数据集的Loss与一般化的数据集（预测数据集）的Loss之间的差异就叫做Generalization error。

而Generalization error又可以细分为Random Error、Bias和Variance三个部分。

首先需要说的是随机误差。它是数据本身的噪声带来的，这种误差是不可避免的。其次如果我们能够获得所有可能的数据集合，并在这个数据集合上将Loss最小化，这样学习到的模型就可以称之为“真实模型”，当然，我们是无论如何都不能获得并训练所有可能的数据的，所以真实模型一定存在，但无法获得，我们的最终目标就是去学习一个模型使其更加接近这个真实模型。

Bias和Variance分别从两个方面来描述了我们学习到的模型与真实模型之间的差距（除去随机误差）。

Bias描述的是对于测试数据集，“用所有可能的训练数据集训练出的所有模型的输出预测结果的期望”与“真实模型”的输出值（样本真实结果）之间的差异。简单讲，就是在样本上拟合的好不好。要想在bias上表现好，low bias，就是复杂化模型，增加模型的参数，但这样容易过拟合 (overfitting)。

Variance则是“不同的训练数据集训练出的模型”的输出值之间的差异。

在一个实际系统中，Bias与Variance往往是不能兼得的。如果要降低模型的Bias，就一定程度上会提高模型的Variance，反之亦然。造成这种现象的根本原因是，我们总是希望试图用有限训练样本去估计无限的真实数据。当我们更加相信这些数据的真实性，而忽视对模型的先验知识，就会尽量保证模型在训练样本上的准确度，这样可以减少模型的Bias。但是，这样学习到的模型，很可能会失去一定的泛化能力，从而造成过拟合，降低模型在真实数据上的表现，增加模型的不确定性。相反，如果更加相信我们对于模型的先验知识，在学习模型的过程中对模型增加更多的限制，就可以降低模型的variance，提高模型的稳定性，但也会使模型的Bias增大。Bias与Variance两者之间的trade-off是机器学习的基本主题之一，机会可以在各种机器模型中发现它的影子。

> [什么是Bias-Variance Tradeoff? - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/142047021)
>
> [机器学习算法系列（18）：方差偏差权衡（Bias-Variance Tradeoff）_passball-CSDN博客](https://blog.csdn.net/passball/article/details/84993600)
>
> [谈谈 Bias-Variance Tradeoff | 始终 (liam.page)](https://liam.page/2017/03/25/bias-variance-tradeoff/)



## END

> 资料部分来源于
>
> [jiqizhixin/Artificial-Intelligence-Terminology: The English-Chinese paired terminologies in Artificial Intelligence Domain (github.com)](https://github.com/jiqizhixin/Artificial-Intelligence-Terminology)
>
> [谷歌开发者机器学习词汇表：纵览机器学习基本词汇与概念 | 机器之心 (jiqizhixin.com)](https://www.jiqizhixin.com/articles/2017-10-05-4)