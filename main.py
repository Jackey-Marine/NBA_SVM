# %% [markdown]
# # predict NBA 🏆 by SVM
# 
# ## 1 package import

# %%
# env
import sys
print(sys.version)

# %%
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

# for visualization 
import matplotlib.pyplot as plt
# for coloring 
import seaborn as sns
# set style of graphs
plt.style.use('classic')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# %% [markdown]
# ## 2 Data cleaning and preprocessing
# 
# We would like to sort the dataset in chronological order, remove the NaN values, and pre-process the data to distinguish between hosts and visitors.

# %%
# load data
df = pd.read_csv("NBA_SVM/data/games.csv")
df.head()

# %%
# # sort data by GAME_DATA_EST
df = df.sort_values(by='GAME_DATE_EST').reset_index(drop = True)
# drop NaN data and check
df = df.loc[df['GAME_DATE_EST'] >= "2004-01-01"].reset_index(drop=True)
df.head()

# %%
# replace Team ID with Names
df_names = pd.read_csv('NBA_SVM/data/teams.csv')
df_names.head()

# %%
# replace 'HOME_TEAM_ID' and 'VISITOR_TEAM_ID' with names in df_names
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
print(df)

# %% [markdown]
# ## 3 Segmentation of Data Set
# 
# We want to try and predict the 2020-2021 NBA play off results starting 2021-08 hence, this portion of the data is the test data set and others are the train data set.

# %%
df = df.loc[df['GAME_DATE_EST'] < '2021-08-01'].reset_index(drop=True)

# %% [markdown]
# ## 4 Features selection
# 
# There are two ways to select the features.
# 1. Select only one feature (points), the prediction is just based on which team has the higher point.
# 2. Select features other than points, the resoult is based on the prediction of a classifier which takes those features as inputs.
# 
# Now we choose the ***second selection***.

# %%
# list all features
feature_list = list(df.columns)
feature_list

# %%
# selecte features
selected_features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    ]
X = df[selected_features]
X.head()

# %%
# check the targets
y = df['HOME_TEAM_WINS']
y.head()

# %%
# turn data into numpy arrays for training
X = X.to_numpy()
y = y.to_numpy()

# %% [markdown]
# ## 5 Fitting SVM

# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.3, random_state = 42)

print("X shape", X_train.shape, "y shape", y_train.shape)

# %%
# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# %%
# train SVM
clf = svm.SVC(kernel='linear') # initialize a model
clf.fit(X_train, y_train) # fit(train) it with the training data and targets

# check test score 
y_pred = clf.predict(X_test) 
print('balanced accuracy score:', balanced_accuracy_score(y_test, y_pred)) 

# %%
# fine-tuning hyperparameters
scoring = make_scorer(balanced_accuracy_score)
param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1,0.1,0.01]}

grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, scoring = scoring, refit=True, verbose=2) 
grid.fit(X_train, y_train)

# %%
# print the best model's hyperparameters
Dis = grid.best_estimator_
print(Dis)

# %% [markdown]
# ## 6 Fitting a Generator
# 
# Since we aim to predict 2020-2021 playoff, here we will just fit the data from that regular session which starts in Oct, 2020.   
# For time-series problems, we give priority to the recent data most reflective of team's recent ability.

# %%
df_ = df.loc[df['GAME_DATE_EST'] > '2020-10-01'].reset_index(drop=True)
df_.head()

# %%
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

# %%
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

# %% [markdown]
# ## 7 Simulation

# %%
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

# %%
class Game:
    
    '''
    
    A game between two teams:
    
    - feature values sampled from Generator
    - win/loss predicted by Discriminator
    
    '''
    
    def __init__ (self, random_state = None):
        
        self.random_state = random_state # keep this to None for making simulations 
    
    def predict(self, team1, team2, num_games = 1):
        
        ''' predict the win or loss of  n game(s) played by two tems'''
        
        assert num_games >= 1, "at least one game must be played"
        # output numpy array
        team_1_feature_data = DATA[team1]
        team_2_feature_data = DATA[team2]
        features = []
        for feature_paras_1 in team_1_feature_data:
            sample_1 = self.sampling(feature_paras_1, size = num_games) # gives a list if num_games> 1
            features.append(sample_1) 
            
        for feature_paras_2 in team_2_feature_data:
            sample_2 = self.sampling(feature_paras_2, size = num_games) # gives a list if num_games> 1
            features.append(sample_2)
            
        features = np.array(features).T 
        win_loss = DIS.predict(features)
        
        return list(win_loss) # a list of win/loss from num_games
    
    
    def sampling(self, dic, size = 1, random_state = None):
        
        '''generate feature values used for making win/loss prediction'''
                        
        dis_name = list(dic.keys())[0] # get the type
        paras = list(dic.values())[0] # get the paras
    
        # get sample
        sample = GEN[dis_name](*paras, size = size,  random_state =  random_state)
            
        return sample 

# %%
class FinalTournament(Game):
    
    ''' Best-of-7 elimination, 16 teams, 4 rounds in total to win championship '''
    
    def __init__(self, n_games_per_group = 7, winning_threshold = 4, random_state = None):

        self.n_games_per_group  = n_games_per_group
        self.winning_threshold = winning_threshold
        self.team_list = None
        self.rounds = {} # keep track the number of times a team wins at each round 
        super().__init__(random_state)
        
    
    def simulate(self, group_list, n_simulation = 1, probs = True):
        
        ''' simulate the entire playoff n times and also record the accumulated wins'''
             
        # update the list of teams
        self.rounds = {}
        self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        
        for i in range(n_simulation):
            cham = self.one_time_simu(group_list)
        if probs:
            self.rounds_probs =  self._compute_probs()
            
    
    def one_time_simu(self, group_list, verbose = False, probs = False):
        
        ''' simulate the entire playoff once and also record the accumulated wins'''
        
        # update the list of teams if haven't done so
        if self.team_list == None: 
            self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        round_number, done = 0, 0
        while not done: 
            all_group_winners, group_list = self.play_round(group_list)
            # retrive round stats
            try:
                updated_round_stats = self.rounds[round_number]
            except KeyError:
                updated_round_stats = {}
                for team in self.team_list:
                    updated_round_stats[team] = 0
            # if a team wins, record + 1 
            for winner in all_group_winners:
                try: 
                    updated_round_stats[winner] += 1
                except KeyError:
                    pass     
            self.rounds[round_number] = updated_round_stats
            if verbose:
                print('{} round played'.format(round_number))
            if probs:
                self.rounds_probs = self._compute_probs()
            if type(group_list) != list: # if it becomes the final
                done = 1
            round_number += 1
            
        return group_list

        
    def play_round(self, group_list):
        
        '''play a round of games based of a list of paired teams'''
        
        all_group_winners = [] 
        # play each group and get the group winner
        for group in group_list:
            winner = self.play_n_games(group[0], group[1])
            all_group_winners.append(winner)
        
        if len(all_group_winners) > 1:
            new_group_list = []         
            for index in range(0, len(all_group_winners), 2):
                # first winner, second winner
                new_group = [all_group_winners[index], all_group_winners[index + 1]]
                new_group_list.append(new_group)
                
            return all_group_winners, new_group_list
        else:  
            return all_group_winners, winner
        
        
    def play_n_games(self, team1, team2):
        
        '''simulate data, and then use our classifier to predict win/loss'''
        
        result = Game().predict(team1, team2, self.n_games_per_group)
        if sum(result[:4]) == self.winning_threshold or sum(result) >= self.winning_threshold:
            winner = team1 # home team wins
        else:
            winner = team2 # visitor team wins
            
        return winner
    
    
    def _compute_probs(self):
        
        '''prob = wins for a team / sum of wins for all teams at a particular round'''
        
        rounds_probs = copy.deepcopy(self.rounds)
        for round_number, round_stats in rounds_probs.items():
            m = np.sum(list(round_stats.values()))
            for k, v in rounds_probs[round_number].items():
                rounds_probs[round_number][k] = v / m
                
        return rounds_probs

# %%
# the below roster is based on 2020-2021 NBA playoffs
# https://en.wikipedia.org/wiki/2020%E2%80%9321_NBA_season

group_list = [
     # Eastern Conference
     ('76ers', 'Wizards'),          # group A 
     ('Knicks', 'Hawks'),           # group B
    
     ('Bucks', 'Heat'),             # group C
     ('Nets', 'Celtics'),           # group D
    
     # Western Conference
     ('Jazz', 'Grizzlies'),         # group E
     ('Clippers', 'Mavericks'),     # group F
    
     ('Nuggets', 'Trail Blazers'),  # group G
     ('Suns', 'Lakers')]            # group H

# %%
# initiate a playoff
playoff = FinalTournament()
# simulate the playoff 5,000 times
playoff.simulate(group_list, n_simulation = 5000)

# %%
# see the winning probabilities from 5,000 playoffs
round_df = pd.DataFrame(playoff.rounds_probs)
round_df

# %% [markdown]
# ## 8 Visualization

# %%
def plotting(rounds_data):
    
    rounds_stats = list(rounds_data.values())
    team_names = list(rounds_stats[0].keys())
    
    # x is number of rounds used for labels, y is a 2-D array of (n_teams, n_rounds) used for data
    x = list(rounds_data.keys())
    y = np.array([list(r.values()) for r in rounds_stats]).T 
    
    # we need at least 16 different colors, one for each team
    c_1 =  sns.color_palette('tab10', n_colors = 10)
    c_2 =  sns.color_palette("pastel", n_colors = 10)
    color_map = c_1 + c_2 
    
    fig = plt.figure()
    plt.stackplot(x, y, labels = team_names, colors = color_map) 
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc = 'upper left', fontsize=13)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Round Number', fontsize = 15)
    plt.title('Winning probabilities by all Teams & Rounds', pad = 20, fontsize = 24)
    plt.tight_layout()
    plt.show()
    
    return fig

# %%
# check that a team's wins should get less and less in later rounds
fig = plotting(playoff.rounds)

# %%
# plot the results: probabilities of winning for all teams at each round
fig = plotting(playoff.rounds_probs)

# %%
# over all rounds winning probabilities
overall_rounds_df = round_df
overall_rounds_df[4] = (round_df[0] + round_df[1] + round_df[2] + round_df[3])/4
overall_rounds_res = overall_rounds_df.sort_values(by=4,ascending=False)
print('Over all rounds winning probabilities:')
print(overall_rounds_res[4].head(5))

# the final round winning probabilities
final_round_res = round_df.sort_values(by=3,ascending=False)
print('The final round winning probabilities:')
print(final_round_res[3].head(5))

# %% [markdown]
# ## 9 Result
# 
# | type | winner |  
# | :-: | :-:|
# | 模型预测结果 | Nets > Nuggets > Bucks |  
# | 实际结果 | Bucks |


