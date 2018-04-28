from __future__ import print_function
import time
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame, Series


inputfile = 'Building_Permits.csv' #输入事务集文件
data = pd.read_csv(inputfile, header=None, dtype = object)

start = time.clock() #计时开始

print(u'\n转换原始数据至0-1矩阵...')

ct = lambda x : pd.Series(1, index = x[pd.notnull(x)]) #转换0-1矩阵的过渡函数

b = list(map(ct, data.as_matrix())) #用map方式执行

data = pd.DataFrame([b]).fillna(0) #实现矩阵转换，空值用0填充

end = time.clock() #计时结束
print(u'\n转换完毕，用时：%0.2f秒' %(end-start))

del b #删除中间变量b，节省内存
