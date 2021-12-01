## ERNIE 1.0

### 亮点

不同的 masking 策略：

1. 基于phrase (在这里是短语比如a series of, written等) 的 masking策略。
2. 基于entity (在这里是人名，位置, 组织，产品等名词比如Apple, J.K. Rowling) 的 masking 策略。

优点：

1. 通过这种方法，在训练过程中隐式地学习了短语和实体的先验知识。相比于 bert 基于字的 mask, 这个单元当中的的所有字在训练的时候，统一被 mask. 对比直接将知识类的 query 映射成向量然后直接加起来，ERNIE 通过统一 mask 的方式可以潜在的学习到知识的依赖以及更长的语义依赖来让模型更具泛化性。
2. ERNIE不是直接添加知识嵌入，而是通过隐式学习知识的相关信息和更长的语义依赖关系，如实体之间的关系、实体的属性和事件的类型，来指导词的嵌入学习。使模型具有较好的泛化和适应性。

## ERNIE 2.0

- 传统的 pre-training 模型主要基于文本中 words 和 sentences 之间的共现进行学习, 训练文本数据中的词法结构，语法结构，语义信息也同样是很重要的。

- 连续学习(Continual Learning)，连续学习的目的是在一个模型中顺序训练多个不同的任务以便在学习下个任务当中可以记住前一个学习任务学习到的结果。通过使用连续学习，可以不断积累新的知识，模型在新任务当中可以用历史任务学习到参数进行初始化，一般来说比直接开始新任务的学习会获得更好的效果。

## 结构

相比 transformer , ERNIE 基本上是transformer 的 encoder 部分，并且 encoder 在结构上是全部一样的，但是并不共享权重，具体区别如下:

Transformer: 6 encoder layers, 512 hidden units, 8 attention heads

ERNIE Base: 12 encoder layers, 768 hidden units, 12 attention heads

ERNIE Large: 24 encoder layers,1024 hidden units, 16 attention heads

从输入上来看第一个输入是一个特殊的CLS, CLS 表示分类任务

就像transformer的一般的encoder, ERINE将一序列的words输入到encoder中。每层使用self-attention, feed-word network,然后把结果传入到下一个encoder



![image-20211013171411668](img\ernie.png)