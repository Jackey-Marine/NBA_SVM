# ------------------------------ #
# 本代码使用SVM的方法去预测NBA总冠军
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

df = pd.read_csv("NBA_SVM/data/teams.csv")
df.head()
