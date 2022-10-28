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

# load data
df = pd.read_csv("NBA_SVM/data/games.csv")
df_names = pd.read_csv('NBA_SVM/data/teams.csv')

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

# Segmentation of Data Set and Select Features
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

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.3, random_state = 42)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# fine-tuning hyperparameters
scoring = make_scorer(balanced_accuracy_score)
param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1,0.1,0.01]}
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, scoring = scoring, refit=True, verbose=2) 
grid.fit(X_train, y_train)
# print the best model's hyperparameters
Dis = grid.best_estimator_
print(Dis)

# Fitting a Generator
df_ = df.loc[df['GAME_DATE_EST'] > '2020-10-01'].reset_index(drop=True)
# define the list of common distributions for fitting
selected_distributions = [
    'norm','t', 'f', 'chi', 'cosine', 'alpha', 
    'beta', 'gamma', 'dgamma', 'dweibull',
    'maxwell', 'pareto', 'fisk']

# extract all the unique teams
unique_teams = df['HOME_TEAM_ID'].unique()

# Get all the data for teams
all_team_sim_data = {}

for team_name in unique_teams:
    
    # find games where the team is either the host or guest
    df_team = df_.loc[(df_['HOME_TEAM_ID'] == team_name) | (df_['VISITOR_TEAM_ID'] == team_name)]
    df_1 = df_team.loc[df_team['HOME_TEAM_ID'] == team_name][selected_features[:5]]
    df_0 = df_team.loc[df_team['VISITOR_TEAM_ID'] == team_name][selected_features[5:]]

    # combine df_0 and df_1
    df_0.columns = df_1.columns
    df_s = pd.concat([df_1, df_0], axis = 0)
    
    # convert the pandas.DataFrame to numpy array
    all_team_sim_data[team_name] = df_s.to_numpy()

megadata = {} # store the data that our Generator will rely on

for team_name in unique_teams:
    
    feature_dis_paras = []
    data = all_team_sim_data[team_name]
    
    # 5 features for each team
    for i in range(5): 
        f = Fitter(data[:, i])
        f.distributions = selected_distributions
        f.fit()
        best_paras = f.get_best(method='sumsquare_error')
        feature_dis_paras.append(best_paras)
        
    megadata[team_name] = feature_dis_paras
    
print('Features for all teams have been fitted!')

# Sim
DATA = megadata.copy() # data that Generator must rely on

GEN = {
 'alpha': stats.alpha.rvs,
 'beta': stats.beta.rvs,
 'chi': stats.chi.rvs,
 'cosine': stats.cosine.rvs,
 'dgamma': stats.dgamma.rvs,
 'dweibull':stats.dweibull.rvs,
 'f':stats.f.rvs,
 'fisk':stats.fisk.rvs,
 'gamma': stats.gamma.rvs,
 'maxwell':stats.maxwell.rvs,
 'norm':stats.norm.rvs,
 'pareto':stats.pareto.rvs,
 't':stats.t.rvs,
}

# feature scaler + fine-turned SVM 
DIS = make_pipeline(scaler, Dis)

