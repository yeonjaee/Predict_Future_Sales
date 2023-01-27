#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# 데이터 로드
import pickle

with open("./data/new2_train_1.pickle","rb") as f:
    matrix = pickle.load(f)
matrix


# ## 함수 정의

# In[7]:


# train, test 셋 분리
x_train = matrix[matrix.date_block_num < 33].drop(columns = 'item_cnt_month')
x_test = matrix[matrix.date_block_num == 33].drop(columns = 'item_cnt_month')
y_train = matrix[matrix.date_block_num < 33].item_cnt_month
y_test = matrix[matrix.date_block_num == 33].item_cnt_month


# In[103]:


# 산점도 그리는 함수
def plot_predictions(name, pred, actual):
    df = pd.DataFrame({'prediction': pred, 'actual': y_test})
    df = df.sort_values(by='actual').reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()



# In[98]:
    
# alpha = 0.01로 적용
lasso = Lasso(alpha=0.01, max_iter=3000)
lasso.fit(x_train, y_train)
pred = lasso.predict(x_test)

mse = mean_squared_error(pred, y_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
print('r2 = {:.3f}, mse = {:.3f}, rmse = {:.3f}'.format(r2, mse, rmse))


plot_predictions(Lasso(0.01), pred, y_test)


# In[74]:


# feature 별 계수(coefficient)
coef_df = pd.DataFrame(index = x_train.columns)
coef_df['coefficient'] = lasso.coef_
coef_df


# In[78]:


# 계수가 0인 feature
coef_df[lasso.coef_==0]


# In[79]:
    
# 계수가 0인 feature들 list에 저장
drop_list = x_test.columns[lasso.coef_==0]
drop_list


# In[80]:


len(drop_list)


# In[95]:
    
# alpha 값 바꿔가면서 성능 비교
alphas = [1, 0.1, 0.01, 0.001, 0.0001]
data = [x_train, x_test, y_train, y_test]


# In[108]:
    
# 결과 출력하는 함수 생성
def lasso_result(alpha, data):
    x_train, x_test, y_train, y_test = data[0], data[1], data[2], data[3]
    
    lasso = Lasso(alpha=alpha, max_iter = 3000)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    
    coef_df = pd.DataFrame(index = x_train.columns)
    coef_df['coefficient'] = lasso.coef_
    coef_df
    
    
    print('Lasso(alpha={})\n'.format(alpha))
    print('r2 = {:.3f}'.format(r2))
    print('mse = {:.3f}'.format(mse))
    print('rmse = {:.3f}'.format(rmse))
    plot_predictions(Lasso(alpha), pred, y_test)
    


# In[109]:


for alpha in alphas:
    lasso_result(alpha, data)


# In[110]:
    
# test data(2015년 11월) 예측
with open("./data/new2_test_1.pickle","rb") as f:
    test_matrix = pickle.load(f)
test_matrix


# In[111]:


x_sub = test_matrix.drop(columns = 'item_cnt_month')


# In[113]:

# 학습 결과에서 가장 결과가 좋았던 alpha=0.0001로 선택해 예측
lasso = Lasso(alpha=0.0001, max_iter=3000)
lasso.fit(x_train, y_train)
pred_sub = lasso.predict(x_sub)
pred_sub


# In[114]:


# 제출 파일 만들기
test_pred = test_matrix[['shop_id', 'item_id']]
test_pred['item_cnt_month_log'] = pred_sub

# 예측값에 log scailing을 했으므로 다시 원래 값으로 수정
test_pred['item_cnt_month'] = np.expm1(pred_sub) 



# In[44]:

# 제출 형식에 맞춰 파일 생성
test_ID = pd.read_csv('./data/test.csv')

m = pd.merge(test_ID, test_pred)


# In[95]:


sub_data = m[['ID', 'item_cnt_month']]

# 20 넘어가는 경우 clip
sub_data['item_cnt_month'] = np.clip(sub_data.item_cnt_month,0,20)


# In[101]:


# 제출파일 csv로 저장
sub_data.to_csv('./kaggle제출/sub_lasso_0302.csv', index=False)


