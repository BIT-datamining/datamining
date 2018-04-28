from __future__ import print_function
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame, Series


# 频繁规则的产生
# 用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i:sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i,len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
    return r
 
def find_rule(data, support, confidence):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    support_series = round(1.0 * data.sum() / len(data), 6)  # 支持度序列
    
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    
    k = 0
    while len(column) > 1:
        k += 1
        print("正在搜索{}元项集".format(k))
        
        column = connect_string(column)
        
        # 通过相乘得1 确认项集结果
        sf = lambda i: data[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数
       
        # 用map函数分布式求解，加快运算速度
        
        data_new = pd.DataFrame(list(map(sf, column)), index=['-'.join(i) for i in column]).T
        
        # 计算占比（支持度）
        support_series_new = round(data_new[['-'.join(i) for i in column]].sum() / len(data),6)
        
        # 通过支持度剪枝
        column = list(support_series_new[support_series_new > support].index)
        support_series = support_series.append(support_series_new)
        
        column_new = []

        for i in column:
            i = i.split('-')
            for j in range(len(i)):
                column_new.append(i[:j] + i[j + 1:] + i[j:j+1])
                
        # 先行定义置信度序列，节约计算时间
        cofidence_series = pd.Series(index=['-'.join(i) for i in column_new])

        for i in column_new:  # 计算置信度序列
            cofidence_series['-'.join(i)] = support_series['-'.join(sorted(i))] / support_series['-'.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series['-'.join(sorted(i.split('-')))]
            
    return result.T

def trans_index(x):
    rename = []
    for item in x.split('-'):
        rename.append(origin_columns[int(item)])
    return "-".join(rename)

if __name__ == '__main__':
    inputfile = 'Building_Permits.csv'
    outputfile = 'income.csv'
    data = pd.read_csv(inputfile)
#    del data['Unnamed: 0']
    # 保存原列标
    origin_columns = list(data.columns)
    # 替换列标
    data.columns = np.arange(0, len(origin_columns)).astype(str)
    # 设置最小支持度
    support = 0.3
    # 设置最小置信度
    confidence = 0.5 
    result = find_rule(data, support, confidence).reset_index()
    # 将关系名称换回原来的
    result['index'] = result['index'].apply(trans_index)
    result = result.set_index('index')
    print("*"*20, "结果", "*"*20)
    print(result)
    result.to_csv(outputfile)
