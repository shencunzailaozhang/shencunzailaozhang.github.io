---
layout: post
title: BiLSTM+CRF笔记
date: 2019-04-04
--- 
BiLSTM+CRF是2018年在英语NER任务上表现最好的模型，学它也有十多天了，之前只是了解个大概，然后忙于学其它东西，结果泛学让我有种什么都没学到的感觉，涉及到内部的还是一无所知，所以今天认真看了它的公式，理解得更深刻了一些。 结构简图如下：  
![](/assets/images/1.png)   
![](/assets/images/2.png)  
# 流程
首先输入单词，单词经过look-up层（通过CBOW、skip-gram或者是Glove等模型构造出look-up table，再将字映射为向量），变成了字向量，再经过BiLSTM层后得到包含上下文信息的向量h，h是上文信息向量h-before和下文信息向量h-after的拼接。再经过一个dropout层后，将h的维度映射为维度为标签个数的向量。再经过一个CRF层后输出得分最高的标签序列，就是我们要求的序列。 
# 公式 
***
假设输入句子如下图（括号中的元素为句中的字）：  
![](/assets/images/3.png)   
对应的标签如下：  
![](/assets/images/4.png)  
我们定义X -> y得到的分数公式如下图：  
![](/assets/images/5.png)   
A的解释如下：  
![](/assets/images/6.png)  
P的解释如下：  
![](/assets/images/10.png) 
![](/assets/images/7.png)  
我们的目标就是输入X对应正确的序列y的概率最大，用似然函数表示如下：  
![](/assets/images/8.png)  
公式的解释和预测如下图：  
![](/assets/images/9.png) 
***  
# 我的理解  
![](/assets/images/11.JPG)  
![](/assets/images/12.JPG)   

***  
以上图片来自一篇[blog](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)和[论文](https://arxiv.org/abs/1603.01360)
