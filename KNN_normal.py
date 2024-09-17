import numpy as np
import math
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

#Train_number为训练数量
Train_number=1

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
        
#邻居类,用来存测试点的相对于点other的距离和other的label
class Neigh:
    #dist距离,label所属的类
    def __init__(self,dist,label):
        self.dist = dist
        self.label = label
        
    def __lt__(self,other):
        return self.dist<other.dist

#计算点a和点b的相对距离,采用欧氏距离
def calc(a,b):
    A=np.array(a)
    B=np.array(b)
    C=A-B
    C=C**2
    return math.sqrt(C.sum())

#拿到test点的分类(仅对于1个点)
def Get_K(Neighbors):
    umap = dict()
    #找前5个点,存取其label
    for i in range(5):
        label=Neighbors[i].label
        label=int(label)
        if label in umap:
            umap[label]+=1
        else:
            umap[label]=1
    #取最大的label
    res=0
    maxx=0
    for (k,v) in umap.items():
        if v>maxx:
            maxx=v
            res=k
    return res
        
#拿到对于test的分类
def get_Predict(data,label,test):
    Neighbors=list()
    #Neighbors用于存储对于点test[j]的邻居
    Res=list()
    for j in range(len(test)):
        if (j+1)%400==0:
            print("已完成",(j+1)/len(test)*100,"%")
        for i in range(len(data)):
            Neighbors.append(Neigh(calc(test[j],data[i]),label[i]))
        Neighbors.sort()
        #对邻居进行排序，方便Get_K函数拿到前5个最近的点
        Res.append(Get_K(Neighbors))
        Neighbors.clear()
    return Res    
    
#检测函数,计算分类准确度
def judge(A,B):
    cnt=0
    for i in range(len(A)):
        if A[i]==B[i]:
            cnt=cnt+1
    print("准确率:",cnt/len(A))
        
#主函数
def __main__():
    print("本次训练集个数:",Train_number)
    train_data=list()
    train_label=list()
    test_data=list()
    test_label=list()
    test_predict_label=list()
    for i in range(Train_number):
        #获取数据
        A,B,C,D=get_Data(train_index[i],test_index[i])
        print("Sucessfully load dataset ",i+1)
        #拿到对于测试集C的预测
        E=get_Predict(A,B,C)
        print("Sucessfully get_predict for dataset ",i+1)
        test_predict_label.append(E)
        train_data.append(A)
        train_label.append(B)
        test_data.append(C)
        test_label.append(D)
        #测试准确度
        judge(D,E)
    print("Complete")

__main__()