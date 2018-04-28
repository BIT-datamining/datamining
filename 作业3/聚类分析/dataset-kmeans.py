
from sklearn.cluster import KMeans
from pandas import DataFrame, Series
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from numpy import *  

#先创建一个900x2 的全零方阵A，并且数据的类型设置为float浮点型  
from numpy import *  
A = zeros((900,2),dtype=float)    

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

#测试集数据格式整理
def txtFileProduce2(f, data_test):
    data_test0 = pd.read_csv('gender_submission.csv')
    with open(f, 'w') as file:
        for i in range(len(data_test)):
            file.write(str(data_test0.Survived[i]) + ' ' + '1 ' + str(data_test.Age[i])
                       + ' ' + '2 ' + str(data_test.Sex[i])
                       + ' ' + '3 ' + str(data_test.Embarked[i]) + '\n')


#训练集数据格式整理
def txtFileProduce1(f, data_train):
    with open(f, 'w') as file:
        for i in range(len(data_train)):

            #data.Survived和data['Survived']此处等价
            #python要求读写文件的数据是字符串
            file.write(str(data_train.Pclass[i])
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
         A[A_row:] = list[0:4]                    #把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行  
         A_row+=1                                #然后方阵A的下一行接着读  

    dataSet=A

    #n_clusters=4，参数设置需要的分类这里设置成4类
    kmeans = KMeans(n_clusters=4, random_state=0).fit(dataSet)
 
    #center为各类的聚类中心，保存在df_center的DataFrame中给数据加上标签
    center=kmeans.cluster_centers_

    df_center = pd.DataFrame(center,columns=['x','y'])

    #标注每个点的聚类结果

    labels=kmeans.labels_

    #将原始数据中的索引设置成得到的数据类别，根据索引提取各类数据并保存

    df = pd.DataFrame(dataSet,index=labels,columns=['x','y'])

    df1 = df[df.index==0]

    df2 = df[df.index==1]

    df3 = df[df.index==2]

    df4 = df[df.index==3]

    #绘图

    plt.figure(figsize=(10,8), dpi=100)

    axes = plt.subplot()

    #s表示点大小，c表示color，marker表示点类型

    type1 = axes.scatter(df1.loc[:,['x']],df1.loc[:,['y']], s=50, c='red',  marker='d')

    type2 = axes.scatter(df2.loc[:,['x']],df2.loc[:,['y']], s=50, c='green',marker='*')

    type3 = axes.scatter(df3.loc[:,['x']],df3.loc[:,['y']], s=50, c='brown',marker='p')

    type4 = axes.scatter(df4.loc[:,['x']],df4.loc[:,['y']], s=50, c='black')

    #显示聚类中心数据点

    type_center = axes.scatter(df_center.loc[:,'x'], df_center.loc[:,'y'], s=40, c='blue')

    plt.xlabel('x',fontsize=16)

    plt.ylabel('y',fontsize=16)

    #显示图例(loc设置图例位置)

    axes.legend((type1, type2, type3,type4,type_center), ('0','1','2','3','center'),loc=1)

    plt.show()
    


if __name__ == '__main__':
    
    function()

    

    

