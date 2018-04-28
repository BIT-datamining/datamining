import matplotlib as plt
from pylab import *
import collections
import copy
import pdb
import numpy as np
import pandas as pd  
from scipy.spatial.distance import cdist

#先创建一个900x3的全零方阵A，并且数据的类型设置为float浮点型  
A = zeros((900,3),dtype=float)   

def preDataHandle(f):
    data_train = pd.read_csv(f)

    #把年龄的缺失值用现有数据的中位数来替代并赋值给原数据
    data_train['Age'] =  data_train['Age'].fillna(data_train['Age'].median())

    #把男性置0，女性置1
    data_train.loc[data_train['Sex'] == 'male', 'Sex'] = 0
    data_train.loc[data_train['Sex'] == 'female', 'Sex'] = 1

    
    #把登船口的缺失值用S替代（整体数据S偏多），并将所有登船口转换为数字
    data_train['Embarked'] = data_train['Embarked'].fillna('S')
    data_train.loc[data_train['Embarked'] == 'S', 'Embarked'] = 0
    data_train.loc[data_train['Embarked'] == 'C', 'Embarked'] = 1
    data_train.loc[data_train['Embarked'] == 'Q', 'Embarked'] = 2
       
    #缺省值补0
    data_train = data_train.fillna(value=0)

    #去除乘客姓名、船票信息和客舱编号三个不打算使用的列   
    data_train=data_train.drop(['Name','Ticket','Cabin'],axis=1)  

    #数据转变为int型

    data_train['Age']=np.array(data_train['Age'],dtype=np.int)

    data_train['Age']=pd.DataFrame(data_train['Age'])

    #返回处理后的数据集
    return data_train


#训练集数据格式整理
def txtFileProduce1(f, data_train):
    with open(f, 'w') as file:
        for i in range(len(data_train)):

            #data.Survived和data['Survived']此处等价
            #python要求读写文件的数据是字符串
            file.write(str(data_train.Survived[i])
                       + ' ' + str(data_train.Pclass[i])
                       + ' ' + str(data_train.Age[i]) + '\n')
  


def function():
    #数据的第一次整理
    data_train = preDataHandle('train.csv')
    


    #数据的第二次整理，生成txt文件
    txtFileProduce1('trained.txt', data_train)


    f = open('trained.txt')         #打开数据文件文件  
    lines = f.readlines()           #把全部数据文件读到一个列表lines中  
    A_row = 0                       #表示矩阵的行，从0行开始  
    for line in lines:              #把lines中的数据逐行读取出来  
         list = line.strip('\n').split(' ')      #处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，
                                     #然后把处理后的行数据返回到list列表中  
         A[A_row:] = list[0:4]       #把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行  
         A_row+=1                    #然后方阵A的下一行接着读  

    dataSet=A

    return dataSet


'''data是一个array, 每一行是一个数据点.下面这个函数计算total cost 根据当前的情况:'''

def total_cost(data, medoids):
    '''
根据当前设置计算总成本。
'''
    med_idx = medoids[-1];
    k = len(med_idx);
    cost = 0.0;

    med = data[ med_idx]
    dis = cdist( data, med, 'euclidean')
    cost = dis.min(axis = 1).sum()
    
    medoids[-2] = [cost]

    
#clustering()函数 分配每一点的归属 根据当前的情况.
def clustering(data, medoids):
    '''
根据当前的medoidids中心和欧几里得距离来计算每个数据点的归属。
'''
    
    # pdb.set_trace()
    med_idx = medoids[-1]
    med = data[med_idx]
    k = len(med_idx)
    

    dis = cdist(data, med)
    best_med_it_belongs_to = dis.argmin(axis = 1)
    for i in range(k):
        medoids[i] =where(best_med_it_belongs_to == i)
        
#kmedoids() 函数
def kmedoids( data, k):

    N = len(data)
    cur_medoids = {}
    cur_medoids[-1] = list(range(k))
    clustering(data, cur_medoids)
    total_cost(data, cur_medoids)
    old_medoids = {}
    old_medoids[-1] = []
    
    iter_counter = 1
    # stop if not improvement.
    while not set(old_medoids[-1]) == set(cur_medoids[-1]):
        print('iteration couter:' , iter_counter)
        iter_counter = iter_counter + 1
        best_medoids = copy.deepcopy(cur_medoids)
        old_medoids = copy.deepcopy(cur_medoids)
        # pdb.set_trace()
        # iterate over all medoids and non-medoids
        i=0
        j=0
        for i in range(N):
            for j in range(k):
                if not i ==j :
                    # swap only a pair
                    tmp_medoids = copy.deepcopy(cur_medoids)
                    tmp_medoids[-1][j] = i

                    clustering(data, tmp_medoids)
                    total_cost(data, tmp_medoids)
                    # pick out the best configuration.
                    if( best_medoids[-2] > tmp_medoids[-2]):
                        best_medoids = copy.deepcopy(tmp_medoids)
        cur_medoids = copy.deepcopy(best_medoids)
        print('current total cost is ', cur_medoids[-2])
    return cur_medoids

if __name__ == '__main__':
    
    data = function()

    # need to change if more clusters are needed .
    k = 3
    medoids = kmedoids(data, k)

    # plot different clusters with different colors.
    scatter( data[medoids[0], 0] ,data[medoids[0], 1], c = 'r')
    scatter( data[medoids[1], 0] ,data[medoids[1], 1], c = 'g')
    scatter( data[medoids[2], 0] ,data[medoids[2], 1], c = 'y')
    scatter( data[medoids[-1], 0],data[medoids[-1], 1] , marker = 'x' , s = 500)
    show()
    

    

