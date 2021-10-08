资源：

[吴恩达《深度学习》作业线上版 - 知乎](https://zhuanlan.zhihu.com/p/95510114)



[xmu-xiaoma666/External-Attention-pytorch: 🍀 Pytorch implementation of various Attention Mechanisms, MLP, Re-parameter, Convolution, which is helpful to further understand papers.⭐⭐⭐](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

[MuQiuJun-AI/bert4pytorch: 超轻量级bert的pytorch版本，大量中文注释，容易修改结构，持续更新](https://github.com/MuQiuJun-AI/bert4pytorch?continueFlag=e9813f68c7afd52a2d0bc48ec4ad1ab1)

[提高科研论文写作效率的小工具 - 知乎](https://zhuanlan.zhihu.com/p/34838403)



## 书

[预训练语言模型 (豆瓣)](https://book.douban.com/subject/35458428/)

[基于深度学习的自然语言处理 (豆瓣)](https://book.douban.com/subject/30236842/)

[NLP简介 - Science is interesting.](https://looperxx.github.io/NLP%E7%9A%84%E5%B7%A8%E4%BA%BA%E8%82%A9%E8%86%80/#46-bert)



## Papers

[thunlp/PLMpapers: Must-read Papers on pre-trained language models.](https://github.com/thunlp/PLMpapers)

[datawhalechina/learn-nlp-with-transformers: we want to create a repo to illustrate usage of transformers in chinese](https://github.com/datawhalechina/learn-nlp-with-transformers)

[datawhalechina/daily-interview: Datawhale成员整理的面经，内容包括机器学习，CV，NLP，推荐，开发等，欢迎大家star](https://github.com/datawhalechina/daily-interview)






我经历了NLP的纯自学过程，我个人觉得算是比较快捷的方法。

其中，也把中文资料和斯坦福的CS224N课程相结合，对NLP做了一个入门学习和简单深入，推荐看看我写的这篇总结文章



## Re-invent Bert系列

| 年份 | 论文标题                                                     | 关键词                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2014 | Memory Network                                               | Memory                                                       |
| 2015 | Neural Machine Translation by Jointly Learning to Align and  Translate Attention | Attention                                                    |
| 2015 | Effective Approaches to Attention-based Neural Machine  Translation | use LSTM                                                     |
| 2015 | End-To-End     Memory Networks                               | multi step attention(k-hop as RNN, global memory)            |
| 2016 | Google’s     Neural Machine Translation System               | low-precision, wordpieces                                    |
| 2016 | A Convolutional Encoder Model for Neural     Machine Translation | CNN encoder                                                  |
| 2017 | Language Modeling with Gated Convolutional Networks          | CNN & GLU(compared with self-attention)                      |
| 2017 | Convolutional     Sequence to Sequence Learning              | Multiple stack, parallel                                     |
| 2017 | Attention is all you need                                    | http://jalammar.github.io/illustrated-transformer/     http://nlp.seas.harvard.edu/2018/04/03/attention.html |
| 2018 | Deep contextualized word representations                     | ELMo                                                         |
| 2018 | Improving Language Understanding     by Generative Pre-Training | GPT                                                          |
| 2018 | BERT: Pre-training of Deep Bidirectional Transformers for Language  Understanding | Bert                                                         |
| 2018 | Language Models are Unsupervised Multitask Learners          | GPT-2                                                        |
| 2019 | RoBERTa: A Robustly Optimized BERT Pretraining Approach      | RoBERTa                                                      |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |







## 周计划

7月22日 - 7月23日两天 入门：

- 了解 nlp 的传统方法和基本概念。

- 掌握 tokenization（分词）方法。
- 了解 jieba, spacy 分词方法，跑例子。
- 了解命名实体识别、情感分析、关系提取的基本过程与概念。

第二周 词向量：

- 序列标注模型， CRF
- 知道常用的词嵌入方法。

- 了解 Word2Vec 模型。
- 利用CRF 、Word2Vec  模型实现对应功能，跑例子。

第三周 词向量：

- 了解 count based global matrix factorization 。
- 了解 glove 模型。
- 利用上述模型实现对应功能，跑例子。

第四周 子词模型：

- 了解 n-gram 思想和 FastText 模型。
- 利用上述模型实现对应功能，跑例子。

第五周 上下文词嵌入模型：

- 了解 contextual word representation
- 了解 ELMO，GPT 与 BERT 模型
- 利用上述模型实现对应功能，跑例子。





## 其他

**中文词向量的探索**

- 练习任务
- 特征词转化为 One-hot 矩阵
- 特征词转化为 tdidf 矩阵
- 利用 word2vec 进行 词向量训练
- word2vec 简单应用
- 利用 one-hot 、TF-idf、word2vec 构建句向量，然后 采用 朴素贝叶斯、条件随机场做分类



传统算法

- 编辑距离：指两个字符串之间，由一个转成另一个所需的最少编辑操作次数
- 集合度量特征：基于BOW(bag of words)，利用集合相似性度量方法，如Jaccard
- 统计特征：如句子长度、词个数、标点数量、标点类型、词性顺序等
- 词向量：将两个文本表示成同一向量空间中的向量，计算欧式距离、余弦相似度等
- 利用TF/IDF/LDA表示进行度量：如BM25文本相似度

其中基于特征的方法，效果很依赖特征的设计。基于传统检索模型的方法存在一个固有缺陷，就是检索模型智能处理Query与Document有重合词的情况，无法处理词语的语义相关性。



深度算法

- Siamese Network
- DSSM（Deep Semantic Structured Model）
- CDSSM（Convolutional Deep Semantic Structured Model）
- ARC-I: Convolutional Neural Network Architectures for Matching Natural Language Sentences
- RNN（Recurrent Neural Networks）
- RNN变种：LSTM、Match-LSTM、Seq-to-Seq、Attention
- DeepMatch
- ARC-II
- MatchPyramid
- Match-SRNN