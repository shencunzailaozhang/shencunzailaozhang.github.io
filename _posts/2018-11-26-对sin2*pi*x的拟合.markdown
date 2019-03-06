---
layout: post
title: 对sin2*pi*x的拟合
date: 2018-11-26
---
# 题目
- 输入为0-1区间内步长为0.01的数x
- 给出的输出要尽可能拟合sin2*pi*x
- 神经网络层数为3，
- 输入层节点为1，隐藏层节点为25，输出层节点为1
- 隐藏层的激活函数为sigmoid函数，输出层激活函数为y=x
# 思路
刚拿到这个题目时候我觉得直接把以前做过的识别手写集的神经网络改一改就可以用了，于是改了，但是怎么也拟合不了，后来才发现之前的网络隐藏层和输出层的激活函数都是sigmoid函数，于是我把公式推导了一遍，但还是无法拟合，我以为这个题目根本做不出来，因为身边的人也没做出来，于是便放弃了，但是过了几天有个同学说做出来了还给我看了拟合情况，还给我讲了一下怎么做的，但我还是没理解，然后看了她的代码，明白了我的问题所在。第二天又看到别人用另一种方法做了，看了很久，看懂了，然后我刚刚又把之前我自己的网络改了一下，也该出来了，挺开心的。以后也不能逃避问题，要直面它，得尽全力之后才能说放弃。
# 关键点
- 本次实验的样本可以是0-1整个区间也可以是其中的数，所以可以输入一个值，也可以输入一个向量，不过输出也要与输入相对应。
- 学习率真的很重要！！！可以从很小的数字开始试，不要因为自己的经验而限制自己行动。
- 要细心，对待权重矩阵的更新要慎重，而且偏置项真的很重要！！！
# 代码一
输入值为一个数，输入了101个数便更新了101次权重和偏置，将它们分别加起来然后求平均值，把平均值矩阵作为最终的权重和偏置矩阵。代码如下：
``` Python
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

I = 0.03  # 学习速率
sampleNum = 101  # 采样点数目
hideCellsNum = 25  # 隐藏单元个数
iterTimes = 500  # 迭代次数
x = np.arange(0, 1.01, 0.01).reshape(sampleNum, 1)
y = 200 * np.sin(2 * np.pi * x)
random=np.random.RandomState(0)
W = random.normal(0.0, pow(1, -0.5), (hideCellsNum,1))
V = random.normal(0.0, pow(1, -0.5), (hideCellsNum,1))
b = random.normal(0.0, pow(1, -0.5), (hideCellsNum,1))
e = random.normal(0.0, pow(1, -0.5), (1,1))
def logistic(t):
    return 1/(1 + np.exp(-t))
sum_loss_list = []
for i in range(iterTimes):  # 第i次迭代
    y_pred = []
    delta_V_all = np.zeros((hideCellsNum, 1))
    delta_W_all = np.zeros((hideCellsNum, 1))
    delta_e_all = np.zeros((1, 1))
    delta_b_all = np.zeros((hideCellsNum, 1))
    loss_list = []
    for k in range(sampleNum):
        # 前向传播
        net1 = W * x[k] + b
        out1 = logistic(net1)
        net2 = np.dot(np.transpose(V), out1)
        out = net2  # 输出激活函数
        y_pred.append(out)
        # loss = float(1.0 / 2 * (y[k] - out) * (y[k] - out))
        loss = 1.0 / 2 * (y[k] - out) * (y[k] - out)
        loss_list.append(loss)

        # 反向传播
        delta_V = -I * (y[k] - out) * (-1) * 1 * out1
        delta_W = -I * (y[k] - out) * (-1) * 1 * V * out1 * (1 - out1) * x[k]
        delta_e = -I * (y[k] - out) * (-1) * 1 * 1
        delta_b = -I * (y[k] - out) * (-1) * 1 * V * out1 * (1 - out1) * 1

        delta_V_all = delta_V_all + delta_V
        delta_W_all = delta_W_all + delta_W
        delta_e_all = delta_e_all + delta_e
        delta_b_all = delta_b_all + delta_b
    V = V + delta_V_all / sampleNum
    W = W + delta_W_all / sampleNum
    b = b + delta_b_all / sampleNum
    e = e + delta_e_all / sampleNum
    sum_loss = sum(loss_list)
    sum_loss_list.append(sum_loss)
    print ('iterate'+str(i)+' loss:'+str(sum_loss))
plt.figure()
plt.plot(x, y, 'red')
plt.plot(x, np.array(y_pred).reshape(sampleNum, 1), 'black')
plt.title('BP_iterate:' + str(i))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
# 代码二
输入值为一个向量，也就是把101个数当作整体来进行权重和偏置的更新，这样比刚刚省事很多，但一开始不容易接受，我花了挺长时间才接受这种思想。代码如下：
``` Python
import numpy as np
import matplotlib.pyplot as plt
m=101
x=np.arange(0,1.01,0.01).reshape(1,101)
y=200*np.sin(2*np.pi*x)
random=np.random.RandomState(10)
input_nodes=1
hidden_nodes=25
output_nodes=1
lr=0.0001
w = random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes,1))
v = random.normal(0.0, pow(output_nodes, -0.5), (1,hidden_nodes))
b = random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, 1))
def sigmoid(x):
    return 1/(1+np.exp(-x))
loss_list=[]
def train(x, y):
    global w,v,b
    vo=v
    hidden_input=w*x+b
    hidden_output=sigmoid(hidden_input)
    final_input=np.dot(v,hidden_output)
    final_output=final_input
    output_error=y-final_output
    loss=np.sum(output_error*output_error)
    loss_list.append(loss)
    v += lr *np.dot(output_error,hidden_output.T)
    z = vo.T * (output_error) * (hidden_output* (1-hidden_output))
    w += lr *np.dot(z, x.T)
    b += lr*np.dot(z,np.ones((101,1)))
def query(x):
    global w, v, b
    hidden_input=w*x+b
    hidden_output=sigmoid(hidden_input)
    final_input=np.dot(v,hidden_output)
    final_output=final_input
    return final_output
epochs=20000
if __name__=='__main__':
    for i in range(epochs):
        train(x,y)
        print("第%d次的损失值是:"%(i),loss_list[i])
    x=np.arange(0,1.01,0.01)
    y=200*np.sin(2*np.pi*x)
    d=query(x).reshape(101,1)
    plt.plot(x,y,'blue')
    plt.plot(x,d,'red')
    plt.show()

```
# 代码三
这个就是我在识别手写集网络基础上改的，本来我以为这样不行，但是在我看了上面两种方法之后我又看了一遍，发现这种方法其实和第一种一样，而且不用求平均值，也就是说压根不用求平均值也可以得到最后的权重矩阵，代码如下。
``` Python
import numpy
import scipy.special
import matplotlib.pyplot as plt
%matplotlib inline
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        random = numpy.random.RandomState(0)
        self.wih =random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who =random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.bih =random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.bho =random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.inodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = self.wih*inputs+self.bih
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bho
        final_outputs = final_inputs
        output_errors = targets - final_outputs
        hidden_errors = self.who.T*output_errors
        self.who += self.lr * output_errors*numpy.transpose(hidden_outputs)
        self.bho+=self.lr * output_errors
        self.wih += self.lr * hidden_errors * hidden_outputs * (1.0 - hidden_outputs)*inputs
        self.bih+=self.lr*hidden_errors * hidden_outputs * (1.0 - hidden_outputs)        
        pass
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = self.wih*inputs+self.bih
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bho
        final_outputs = final_inputs
        return final_outputs
input_nodes = 1
hidden_nodes = 25
output_nodes = 1
learning_rate = 0.0003
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
epochs =200000
y_pred=np.zeros(100)
X=numpy.arange(0,1,0.01)
y=200*numpy.sin(2*numpy.pi*X)
j=0
for i in range(epochs):
    n.train(X[j],y[j])
    j=j+1
    if(j%100==0):
        j=0
for i in range(100):
    y_pred[i]=n.query(X[i])
plt.plot(X,y,'blue')
plt.plot(X,y_pred,'red')
plt.show()
```
# 总结
以后也要努力啊，不要逃避问题，尽全力之后才有资格说放弃！！！

