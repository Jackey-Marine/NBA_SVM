# ------------------------------ #
# predict NBA chamption by SVM
# ------------------------------ #

import numpy as np
import pandas as pd

# for classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.pipeline import make_pipeline

# for Generator
from scipy import stats
from fitter import Fitter
import copy

# for Figure
import matplotlib.pyplot as plt  # for visualization 
import seaborn as sns  # for coloring 
# set style of graphs
plt.style.use('dark_background')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# ------------------------------ #
# Data cleaning and preprocessing
# ------------------------------ #

# 我们希望能将数据集按时间顺序排序，并去掉NaN的数据，同时完成主客队区别的数据预处理。

# load data
df = pd.read_csv("NBA_SVM/data/games.csv")
df_names = pd.read_csv('./data/teams.csv')

# sort by time
df = df.sort_values(by='GAME_DATE_EST').reset_index(drop = True)
df = df.loc[df['GAME_DATE_EST'] >= "2004-01-01"].reset_index(drop=True)

# replace 'HOME_TEAM_ID' and 'VISITOR_TEAM_ID'
df_names = df_names[['TEAM_ID', 'NICKNAME']]

# replace 'HOME_TEAM_ID'
home_names = df_names.copy()
home_names.columns = ['HOME_TEAM_ID', 'NICKNAME']
result_1 = pd.merge(df['HOME_TEAM_ID'], home_names, how ="left", on="HOME_TEAM_ID")  
df['HOME_TEAM_ID'] = result_1['NICKNAME']

# replace 'VISITOR_TEAM_ID'
visitor_names = df_names.copy()
visitor_names.columns = ['VISITOR_TEAM_ID', 'NICKNAME']
result_2 = pd.merge(df['VISITOR_TEAM_ID'], visitor_names, how = "left", on="VISITOR_TEAM_ID")
df['VISITOR_TEAM_ID'] = result_2['NICKNAME']

# ------------------------------ #
# Segmentation of Data Set and Select Features
# ------------------------------ #

# 我们想试着预测从2021-08赛季开始的2020-2021赛季NBA季后赛的结果，因此这部分数据是测试数据集,剩余数据为训练集。
# 我们选择分数以外的特征的一部分作为训练集数据，结果基于分类器的预测，分类器将这些特征作为输入。

df = df.loc[df['GAME_DATE_EST'] < '2020-08-01'].reset_index(drop=True)
feature_list = list(df.columns)

# selecte features
selected_features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    ]
X = df[selected_features]
y = df['HOME_TEAM_WINS']

# turn X,y into numpy arrays for training
X = X.to_numpy()
y = y.to_numpy()

