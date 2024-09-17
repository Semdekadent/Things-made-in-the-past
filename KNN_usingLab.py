import numpy as np
import math
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier

Train_number=5

data_mat = sio.loadmat('mnist.mat')

data = data_mat['data']
label = data_mat['label']
label = label.T

index_mat = sio.loadmat('index.mat')

train_index = index_mat['train_index']
test_index = index_mat['test_index']

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
    count=0
    for i in range(len(A)):
        if A[i]==B[i]:
            count=count+1
    print("Accuracy Rate:",count/len(A))

def __main__():
    for i in range(Train_number):
        DA,DAL,TA,TAL=get_Data(train_index[i],test_index[i])
        train_data = np.array(DA)
        train_label = np.array(DAL)
        test_data = np.array(TA)
        test_label = np.array(TAL)
        print("Sucessfully load dataset ",i+1)
        print()

        print("Begin training.")
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(train_data,train_label.ravel())
        score = knn.score(train_data,train_label.ravel())
        print("Successfully trained.")
        print()
        print("Begin predicting.")

        test_res = knn.predict(test_data)
        print("Successfully predicted.\n")

        Judge(test_res,test_label)
        print()

__main__()