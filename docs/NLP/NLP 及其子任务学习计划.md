资源：

[吴恩达《深度学习》作业线上版 - 知乎](https://zhuanlan.zhihu.com/p/95510114)



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