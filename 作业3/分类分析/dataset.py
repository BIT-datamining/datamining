import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
  
dt_train=pd.read_csv('train.csv')  
dt_test=pd.read_csv('test.csv')  #读取csv  

#年龄Age和舱房Cabin存在空值。用“0”补充上，
dt_train = dt_train.fillna(value=0)
dt_test = dt_test.fillna(value=0)

#去除乘客姓名、船票信息和客舱编号三个不打算使用的列   
dt_train_p=dt_train.drop(['Name','Ticket','Cabin'],axis=1)  
dt_test_p=dt_test.drop(['Name','Ticket','Cabin'],axis=1)
'''
#按照性别和舱位分组聚合
Pclass_Gender_grouped=dt_train_p.groupby(['Sex','Pclass'])
 #计算存活率 
PG_Survival_Rate=(Pclass_Gender_grouped.sum()/Pclass_Gender_grouped.count())['Survived']  
  
x=np.array([1,2,3])  
width=0.3  
plt.bar(x-width,PG_Survival_Rate.female,width,color='r')  
plt.bar(x,PG_Survival_Rate.male,width,color='b')  
plt.title('Survival Rate by Gender and Pclass')  
plt.xlabel('Pclass')  
plt.ylabel('Survival Rate')  
plt.xticks([1,2,3])  
plt.yticks(np.arange(0.0, 1.1, 0.1))  
plt.grid(True,linestyle='-',color='0.7')  
plt.legend(['Female','Male'])  
plt.show()  #画图  
'''

dt_train_p['Relatives']=dt_train_p['SibSp']+dt_train_p['Parch']  
Rela_grouped=dt_train_p.groupby(['Relatives'])  
Rela_Survival_Rate=(Rela_grouped.sum()/Rela_grouped.count())['Survived']  
Rela_count=Rela_grouped.count()['Survived']  
  
ax1=Rela_count.plot(kind='bar',color='g')  
ax2=ax1.twinx()  
ax2.plot(Rela_Survival_Rate.values,color='r')  
ax1.set_xlabel('Relatives')  
ax1.set_ylabel('Number')  
ax2.set_ylabel('Survival Rate')  
plt.title('Survival Rate by Relatives')  
plt.grid(True,linestyle='-',color='0.7')  
plt.show()  
