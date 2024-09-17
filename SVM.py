import numpy as np
import math
import scipy.io as sio
from sklearn import svm

#Train_number为训练数量
Train_number=5

#读取数据集矩阵
data_mat = sio.loadmat('mnist.mat')
#分割数据集为数据和label
data = data_mat['data']
label = data_mat['label']
label = label.T

#读取划分矩阵
index_mat = sio.loadmat('index.mat')
#分割划分矩阵为训练矩阵和测试矩阵
train_index = index_mat['train_index']
test_index = index_mat['test_index']

#获取数据,返回测试数据及其label,训练数据及其label
def get_Data(train,test):
    Train=list()
    Train_label=list()
    Test=list()
    Test_label=list()
    for t in train:
        Train.append(data[t])
        Train_label.append(label[t])
    for t in test:
        Test.append(data[t])
        Test_label.append(label[t])
    return Train,Train_label,Test,Test_label

def Judge(A,B):
    cnt=0
    for i in range(len(A)):
        if A[i]==B[i]:
            cnt=cnt+1
    print("Accuracy Rate:",cnt/len(A))


def __main__():
    #分别获取训练和测试数据集及其label
    #并将其转换为矩阵
    for i in range(Train_number):
        DA,DAL,TA,TAL=get_Data(train_index[i],test_index[i])
        train_data = np.array(DA)
        train_label = np.array(DAL)
        test_data = np.array(TA)
        test_label = np.array(TAL)
        print("Sucessfully load dataset ",i+1)
        print()

        #训练模型
        print("Begin training.")
        classifier = svm.SVC(C=2,kernel='linear',gamma=10,decision_function_shape='ovr')
        classifier.fit(train_data,train_label.ravel())
        print("Successfully trained.\n")
        print("Begin to test.")
        print("Training Set",classifier.score(test_data,test_label))
        print()

__main__()