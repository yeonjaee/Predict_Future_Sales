# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:06:53 2022

@author: rkdtk
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from xgboost import plot_importance
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

with open("./data/robust_train2.pickle","rb") as fr:
    data = pickle.load(fr)
    
data.columns
with open("./data/robust_test2.pickle","rb") as fr:
    test = pickle.load(fr)
test = test.drop(['item_cnt_month'], axis=1)

## 데이터 분리
X = data.drop(['item_cnt_month'], axis=1)
y = data['item_cnt_month']

## PCA 주성분 분석
pca = PCA(n_components=5)
printcipalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=printcipalComponents)

# explained_variance_ratio_는 고유값이며 설명가능한 분산량을 의미한다.
# 5개의 주성분으로 99% 이상 설명력을 가지는 것을 확인
pca.explained_variance_ratio_
X = principalDf

# 테스트 데이터도 동일한 5개 차원으로 축소
printcipalComponents = pca.transform(test)
principalDf = pd.DataFrame(data=printcipalComponents)

pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
test = principalDf

## 모델 생성 및 학습
xgb = XGBRegressor(n_estimators=200, max_depth=12, learning_rate=0.05, subsample=0.7,
                   colsample_bytree = 0.7, random_state = 42)
xgb.fit(X, y)
pred = xgb.predict(test)
pred = pd.DataFrame(pred)
pred.columns = ['item_cnt_month']
temp = [i for i in range(0,214200,1)]
pred['ID'] = temp
pred = pred[['ID', 'item_cnt_month']]
pred

pred.to_csv('./kaggle제출/XGBoost_submission.csv', index=False)
print('done')

pd.read_csv('./kaggle제출/XGBoost_submission.csv')
## 피처 importance 확인
plt.rcParams["figure.figsize"] = (16, 10)
plot_importance(xgb, max_num_features=30)