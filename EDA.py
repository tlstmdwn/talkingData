import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from multiprocessing import Pool
import time

## 파일 load
filename = './Desktop/Kaggle_Study/all/train.csv'
num_lines = sum(1 for l in open(filename))
n=10
skip_idx = [x for x in range(1, num_lines) if x % n != 0]
talkingData_train = pd.read_csv(filename, skiprows=skip_idx)
talkingData_valid = pd.read_csv('./Desktop/Kaggle_Study/all/train_sample.csv', skiprows=skip_idx)
talkingData_test = pd.read_csv( './Desktop/Kaggle_Study/all/test.csv')

#데이터 type 명목형으로 변경
variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    talkingData_test[v] = talkingData_test[v].astype('category')
    talkingData_train[v]=talkingData_train[v].astype('category')
    talkingData_valid[v]=talkingData_valid[v].astype('category')
    
#시간 데이터 변경
talkingData_train['click_time'] = pd.to_datetime(talkingData_train['click_time'])
talkingData_train['attributed_time'] = pd.to_datetime(talkingData_train['attributed_time'])
talkingData_valid['click_time'] = pd.to_datetime(talkingData_valid['click_time'])
talkingData_valid['attributed_time'] = pd.to_datetime(talkingData_valid['attributed_time'])

talkingData_test['click_time'] = pd.to_datetime(talkingData_test['click_time'])

#set as_attributed in train as a categorical
talkingData_train['is_attributed']=talkingData_train['is_attributed'].astype('category')
talkingData_valid['is_attributed']=talkingData_valid['is_attributed'].astype('category')

#trainset 변수 종류 개수 
plt.figure(figsize=(15, 8))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(talkingData_train[col].unique()) for col in cols]
sns.set(font_scale=1.2)
pal = sns.color_palette()
ax = sns.barplot(cols, uniques, palette=pal, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
    
    

#testset 변수 종류 개수 
plt.figure(figsize=(15, 8))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(talkingData_test[col].unique()) for col in cols]
sns.set(font_scale=1.2)
pal = sns.color_palette()
ax = sns.barplot(cols, uniques, palette=pal, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


#trainset target인 is_attributed 개수 차이
plt.figure(figsize=(15, 8))
at = talkingData_train['is_attributed'].value_counts().plot(kind='bar')

at.set(ylabel = "App download count", title = 'App download difference')
for p in range(len(at.patches)):
    at_patch = at.patches[p]
    at.text(at_patch.get_x()+at_patch.get_width()/2, at_patch.get_height()+50, talkingData_train['is_attributed'].value_counts()[p],
            ha = 'center')
plt.show()




#trainset 각변수별 거짓 다운로두 수 상위 100개까지
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [talkingData_train[talkingData_train['is_attributed'] ==0][col].value_counts().sort_values(ascending =False)
           [lambda x : x>0][0:100] for col in cols]
sns.set(font_scale=1.2)
for col, uniq in zip(cols, uniques):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax = uniq.plot(kind='bar')
    ax.set(xlabel='{}'.format(col), ylabel='Fraud download count', title='Number of fraud count {}'.format(col))
    plt.show()
    
    
    
  #trainset 각변수별 실제 download 수 상위 100개까지
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [talkingData_train[talkingData_train['is_attributed'] ==1][col].value_counts().sort_values(ascending =False)
           [lambda x : x>0][0:100] for col in cols]
sns.set(font_scale=1.2)
for col, uniq in zip(cols, uniques):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax = uniq.plot(kind='bar')
    ax.set(xlabel='{}'.format(col), ylabel='real download count', title='Number of fraud count {}'.format(col))
    plt.show()  
    
    
    
   #전체 time별 클릭수 확인
plt.figure(figsize=(15,8))
ax = talkingData_train['click_time'].dt.round('H').value_counts().plot(linestyle='-',legend = True)
ax.set(xlabel='time', ylabel='click_count', title='Number of click count by time')

plt.show()




#하루 시간, 날짜, 요일별 클릭 수
time_line = ['%H','%d']
time_line_name = ['hour', 'day']
for name in range(len(time_line)):
    ttt = talkingData_train['click_time'].dt.strftime(time_line[name]).value_counts().sort_index()
    fig.add_subplot(1, 1, 1)
    plt.figure(figsize=(15,8))
    
    plt.plot(ttt.index,ttt.values)
    plt.title("{} click_count".format(time_line_name[name]))
    plt.xlabel("{}".format(time_line_name[name]))
    plt.ylabel('click_count')
    plt.show()
