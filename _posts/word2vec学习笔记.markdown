（说在前面，下面这篇笔记里有很多都是网上的内容，但考虑到没人看，所以我就不注明出处了）
由于最近在做病历处理方面的东西，需要提取病人口述的病症，根据病症预测病人患某种病的概率是多大。为了能让计算机处理病症，需要先将它们向量化，向量化有两种方式
# one-hot向量
虽然onehot向量简单，但是向量维度和词语数量相等，而且因为存在大量的0，会导致浪费计算资源。另一个弊端是它无法反映词语之间的联系，所以转而用分布式表示。
# 分布式表示
分布式表示维度可控，不会因为词语过多而导致维度灾难，而且可以很好地反映词语之间的关系，通过计算欧式距离可以算出词语之间的相近程度。
分布式表示有以下几种
- 基于矩阵的分布式表示
- 基于聚类的分布式表示
- 基于神经网络的分布式表示（词嵌入word embedding）  
主要将基于神经网络的分布式表示，基于神经网络的分布式表示又有以下几种模型
  - Neural Network Language Model ，NNLM
  - Log-Bilinear Language Model， LBL
  - Recurrent Neural Network based Language Model，RNNLM
  - Collobert 和 Weston 在2008 年提出的 C&W 模型
  - Mikolov 等人提出了 CBOW（ Continuous Bagof-Words）和 Skip-gram 模型  
这些和word2vec有什么关系呢？
### word2vec是包含CBOW和Skip-gram的训练词向量的工具。  
所以学word2vec其实是在学CBOW和Skip-gram。CBOW是给定上下文单词来预测中心词，Skip-gram是给定中心词来预测上下文单词，如图所示![图片](/assets/images/structure.jpg)

