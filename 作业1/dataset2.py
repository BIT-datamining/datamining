import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame, Series




'标称属性'
name_category = ['Permit Type', 'Block', 'Lot', 'Street Number', 'Street Number Suffix', 'Street Name', 'Street Suffix', 
              'Current Status', 'Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit',  
              'Existing Use', 'Proposed Use', 'Plansets', 'TIDF Compliance', 'Existing Construction Type', 
              'Proposed Construction Type', 'Site Permit', 'Supervisor District', 'Neighborhoods - Analysis Boundaries']
'数值属性'
name_value = ['Number of Existing Stories', 'Number of Proposed Stories', 'Estimated Cost', 'Revised Cost', 
                 'Existing Units', 'Proposed Units']

dataset_path = 'Building_Permits.csv'

data_origin = pd.read_csv(dataset_path, 
                   na_values='None',
                   low_memory=False)

dropna=False
format_width=30

# 数据可视化和摘要

'标称属性，给出每个可能取值的频数'
format_text = '{{:<{0}}}{{:<{0}}}'.format(format_width)
for col in name_category:
    print('标称属性 <{}> 频数统计'.format(col))
    print(format_text.format('value', 'count'))
    print('- ' * format_width) 
    counts = pd.value_counts(data_origin[col].values, dropna=False)
    for i, index in enumerate(counts.index):
        if pd.isnull(index): # NaN?
            print(format_text.format('-NaN-', counts.values[i]))
        else:
            print(format_text.format(index, counts[index]))
    print('--' * format_width)
    print()


'对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。'

# 最大值
data_show = pd.DataFrame(data = data_origin[name_value].max(), columns = ['max'])
# 最小值
data_show['min'] = data_origin[name_value].min()
# 均值
data_show['mean'] = data_origin[name_value].mean()
# 中位数
data_show['median'] = data_origin[name_value].median()
# 四分位数
data_show['quartile'] = data_origin[name_value].describe().loc['25%']
# 缺失值个数
data_show['missing'] = data_origin[name_value].describe().loc['count'].apply(lambda x : 200-x)

print(data_show)



# 绘图配置


'直方图'
fig = plt.figure(figsize = (20,11))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 2, i)
    data_origin[item].plot(kind = 'hist', title = item, ax = ax)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('histogram.jpg')
print('histogram saved at histogram.jpg')



'qq图'
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 2, i)
    sm.qqplot(data_origin[item], ax = ax)
    ax.set_title(item)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('qqplot.jpg')
print('qqplot saved at qqplot.jpg')

'盒图'
fig = plt.figure(figsize = (20,12))
i = 1
for item in name_value:
    ax = fig.add_subplot(3, 2, i)
    data_origin[item].plot(kind = 'box')
    i += 1
fig.savefig('boxplot.jpg')
print('boxplot saved at boxplot.jpg')


#数据缺失的处理

# 1 将缺失部分剔除

'找出含有缺失值的数据条目索引值'
nan_list = pd.isnull(data_origin).any(1).nonzero()[0]


'显示含有缺失值的原始数据条目'
data_origin.iloc[nan_list].style.highlight_null(null_color='red')

'将缺失值对应的数据整条剔除，生成新数据集'
data_filtrated = data_origin.copy()

data_filtrated.dropna()

'绘制可视化图'
fig = plt.figure(figsize = (20,15))

i = 1
'对标称属性，绘制折线图'
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)    
    i += 1

i = 6
'对数值属性，绘制直方图'
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'filtrated', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

'保存图像和处理后数据'
fig.savefig('missing_data_delete.jpg')
data_filtrated.to_csv('missing_data_delete.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print('filted_missing_data1 saved at missing_data_delete.jpg')
print('data after analysis saved at missing_data_delete.csv')



# 2 用最高频率值来填补缺失值

'建立原始数据的拷贝'

data_filtrated = data_origin.copy()

'对每一列数据，分别进行处理'
for item in name_category+name_value:
    # 计算最高频率的值
    most_frequent_value = data_filtrated[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated[item].fillna(value = most_frequent_value, inplace = True)

'绘制可视化图'
fig = plt.figure(figsize = (20,15))

i = 1
'对标称属性，绘制折线图'
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1    

i = 6
'对数值属性，绘制直方图'
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

'保存图像和处理后数据'
fig.savefig('missing_data_most.jpg')
data_filtrated.to_csv('missing_data_most.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print('filted_missing_data2 saved at missing_data_most.jpg')
print('data after analysis saved at missing_data_most.csv')

# 3 通过属性的相关关系来填补缺失值

'建立原始数据的拷贝'
data_filtrated = data_origin.copy()
'对数值型属性的每一列，进行插值运算'
for item in name_value:
    data_filtrated[item].interpolate(inplace = True)

'绘制可视化图'
fig = plt.figure(figsize = (20,15))

i = 1
'对标称属性，绘制折线图'
for item in name_category:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    pd.value_counts(data_origin[item].values).plot(ax = ax, marker = '^', label = 'origin', legend = True)
    pd.value_counts(data_filtrated[item].values).plot(ax = ax, marker = 'o', label = 'filtrated', legend = True)
    i += 1   
    
i = 10
'对数值属性，绘制直方图'
for item in name_value:
    ax = fig.add_subplot(4, 5, i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

'保存图像和处理后数据'
fig.savefig('missing_data_corelation.jpg')
data_filtrated.to_csv('missing_data_corelation.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print('filted_missing_data3 saved at missing_data_corelation.jpg')
print('data after analysis saved at data_output/missing_data_corelation.csv')


if __name__ == '__main__':
    pass
