import pandas as pd
import numpy as np
import random
import time

#定义神经网络的参数
d=3#输入节点个数
l=1#输出节点个数
q=2*d+1#隐层个数,采用经验公式2d+1
eta=0.5#学习率
error=0.002#精度
train_num=480#训练数据个数
test_num=240#测试数据个数

#初始化权值阈值

w1=[[random.random() for i in range(q)] for j in range(d)]
w2=[[random.random() for i in range(l)] for j in range(q)]
b1=[random.random() for i in range(q)]
b2=[random.random() for i in range(l)]
'''
b2=[0.0 for i in range(l)]
b1=[0.0 for i in range(q)]
#w1 = [[random.normalvariate(0, 1) for i in range(q)] for j in range(d)]
#w2 = [[random.normalvariate(0, 1) for i in range(l)] for j in range(q)]
w1 = [[0.5 for i in range(q)] for j in range(d)]
w2 = [[0.5 for i in range(l)] for j in range(q)]
'''

#读取气温数据
dataset = pd.read_csv('tem.csv', delimiter=",")
dataset=np.array(dataset)
m,n=np.shape(dataset)
totalX=np.zeros((m-d,d))
totalY=np.zeros((m-d,l))
for i in range(m-d):#分组：前三个值输入，第四个值输出
    totalX[i][0]=dataset[i][0]
    totalX[i][1]=dataset[i+1][0]
    totalX[i][2]=dataset[i+2][0]
    totalY[i][0]=dataset[i+3][0]

#归一化数据
Normal_totalX=np.zeros((m-d,d))
Normal_totalY=np.zeros((m-d,l))
nummin=np.min(dataset)
nummax=np.max(dataset)
dif=nummax-nummin
for i in range(m-d):
    for j in range(d):
        Normal_totalX[i][j]=(totalX[i][j]-nummin)/dif
    Normal_totalY[i][0]=(totalY[i][0]-nummin)/dif

#截取训练数据
trainX=Normal_totalX[:train_num-d,:]#训练数据
trainY=Normal_totalY[:train_num-d,:]
testX=Normal_totalX[train_num:,:]#测试数据
testY=Normal_totalY[train_num:,:]
m,n=np.shape(trainX)

#实现sigmoid函数
import math
def sigmoid(iX):
    for i in range(len(iX)):
        iX[i] = 1 / (1 + math.exp(-iX[i]))
    return iX

#网络训练
start = time.clock()#起始时间
iter=0
while True:
    sumE=0
    for i in range(m):#每行循环
        alpha=np.dot(trainX[i],w1)
        b=sigmoid(alpha-b1)
        beta=np.dot(b,w2)
        predictY=sigmoid(beta-b2)
        E = (predictY-trainY[i])*(predictY-trainY[i])
        sumE+=E
        #梯度下降法
        g=predictY*(1-predictY)*(trainY[i]-predictY)
        e=b*(1-b)*((np.dot(w2,g.T)).T)
        w2+=eta*np.dot(b.reshape((q,1)),g.reshape((1,l)))
        b2-=eta*g
        w1+=eta*np.dot(trainX[i].reshape((d,1)),e.reshape((1,q)))
        b1-=eta*e
    sumE=sumE/m
    iter+=1
    if iter % 10 == 0:#每训练10次，输出误差
        print("第 %d 次训练后,误差为：%g" % (iter, sumE))
    if sumE<error:#误差小于0.002，退出循环
        break
print("循环训练总次数：",iter)

end = time.clock()#结束时间
print("运行耗时(s)：",end-start)

#测试,求均方根误差
m,n=np.shape(testX)
MSE=0
for i in range(m):
    alpha = np.dot(testX[i],w1)  
    b=sigmoid(alpha-b1)
    beta = np.dot(b,w2)  
    y=sigmoid(beta - b2)
    #testY[i]=testY[i]*dif+nummin#由于变量的地址传递，存在隐患
    YY=testY[i]*dif+nummin#反归一化
    y=y*dif+nummin
    #MSE=(y-testY[i])*(y-testY[i])+MSE
    MSE=(y-YY)*(y-YY)+MSE
MSE=MSE/m
print("测试集均方误差：",MSE)

#预测
def predict(iX):
    iX=(iX-nummin)/dif#归一化
    alpha = np.dot(iX, w1)  
    b=sigmoid(alpha-b1)
    beta = np.dot(b, w2)  
    predictY=sigmoid(beta - b2)
    predictY=predictY*dif+nummin#反归一化
    return predictY
	
XX=[18.3,17.4,16.7]
XX=np.array(XX)
print("[18.3,17.4,16.7]输入下,预测气温为：",predict(XX))