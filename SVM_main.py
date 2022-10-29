# 1 package import
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

# 2 Data cleaning and preprocessing
# load data
df = pd.read_csv("NBA_SVM/data/games.csv")

# plot total number of games played in each season
fig, ax = plt.subplots()
v_c = df['SEASON'].value_counts().sort_index()
v_c.index = v_c.index.astype(str)
ax.bar(v_c.index, v_c.values)
plt.title("Total number of games played in each season")
plt.show()

# sort data by GAME_DATA_EST
df = df.sort_values(by='GAME_DATE_EST').reset_index(drop = True)
# drop NaN data and check
df = df.loc[df['GAME_DATE_EST'] >= "2004-01-01"].reset_index(drop=True)
# replace Team ID with Names
df_names = pd.read_csv('NBA_SVM/data/teams.csv')
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

# 3 Segmentation of Data Set & Features selection
df = df.loc[df['GAME_DATE_EST'] < '2021-08-01'].reset_index(drop=True)
# list all features
feature_list = list(df.columns)

# The correlation coefficient of the features
corr = df[feature_list].corr()
fig, ax = plt.subplots(figsize=(10, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, vmax=1, center=0, vmin=-1, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# selecte features
selected_features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    ]
X = df[selected_features]
X.head()
# check the targets
y = df['HOME_TEAM_WINS']
y.head()
# turn data into numpy arrays for training
X = X.to_numpy()
y = y.to_numpy()

# 4 Fitting SVM
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.3, random_state = 42)
print("X shape", X_train.shape, "y shape", y_train.shape)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# train SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# check test score 
y_pred = clf.predict(X_test) 
print('balanced accuracy score:', balanced_accuracy_score(y_test, y_pred)) 

# fine-tuning hyperparameters
scoring = make_scorer(balanced_accuracy_score)
param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1,0.1,0.01]}

grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, scoring = scoring, refit=True, verbose=2) 
grid.fit(X_train, y_train)

Dis = grid.best_estimator_
print(Dis)

# 5 Fitting a Generator
df_ = df.loc[df['GAME_DATE_EST'] > '2020-10-01'].reset_index(drop=True)

# define the list of common distributions for fitting
selected_distributions = [
    'norm','t', 'f', 'chi', 'cosine', 'alpha', 
    'beta', 'gamma', 'dgamma', 'dweibull',
    'maxwell', 'pareto', 'fisk']

unique_teams = df['HOME_TEAM_ID'].unique()

all_team_sim_data = {}

for team_name in unique_teams:
    
    df_team = df_.loc[(df_['HOME_TEAM_ID'] == team_name) | (df_['VISITOR_TEAM_ID'] == team_name)]
    df_1 = df_team.loc[df_team['HOME_TEAM_ID'] == team_name][selected_features[:5]]
    df_0 = df_team.loc[df_team['VISITOR_TEAM_ID'] == team_name][selected_features[5:]]

    df_0.columns = df_1.columns
    df_s = pd.concat([df_1, df_0], axis = 0)
    
    all_team_sim_data[team_name] = df_s.to_numpy()

megadata = {}

for team_name in unique_teams:
    
    feature_dis_paras = []
    data = all_team_sim_data[team_name]
    
    # five features for each team
    for i in range(5): 
        f = Fitter(data[:, i])
        f.distributions = selected_distributions
        f.fit()
        best_paras = f.get_best(method='sumsquare_error')
        feature_dis_paras.append(best_paras)
        
    megadata[team_name] = feature_dis_paras
    
print('All features fitted!')

# class
class Game:
    '''
    A game between two teams:
    - feature values sampled from Generator
    - win/loss predicted by Discriminator
    '''
    def __init__ (self, random_state = None):
        
        self.random_state = random_state
    
    def predict(self, team1, team2, num_games = 1):
        
        ''' predict the win or loss of  n game(s) played by two tems'''
        
        assert num_games >= 1, "at least one game must be played"
        
        team_1_feature_data = DATA[team1]
        team_2_feature_data = DATA[team2]
        features = []
        for feature_paras_1 in team_1_feature_data:
            sample_1 = self.sampling(feature_paras_1, size = num_games)
            features.append(sample_1) 
            
        for feature_paras_2 in team_2_feature_data:
            sample_2 = self.sampling(feature_paras_2, size = num_games)
            features.append(sample_2)
            
        features = np.array(features).T 
        win_loss = DIS.predict(features)
        
        return list(win_loss)
    
    
    def sampling(self, dic, size = 1, random_state = None):
        
        '''generate feature values used for making win/loss prediction'''
                        
        dis_name = list(dic.keys())[0]
        paras = list(dic.values())[0]

        sample = GEN[dis_name](*paras, size = size,  random_state =  random_state)
            
        return sample 

class FinalTournament(Game):
    
    ''' Best-of-7 elimination, 16 teams, 4 rounds in total to win championship '''
    
    def __init__(self, n_games_per_group = 7, winning_threshold = 4, random_state = None):

        self.n_games_per_group  = n_games_per_group
        self.winning_threshold = winning_threshold
        self.team_list = None
        self.rounds = {} 
        super().__init__(random_state)
        
    
    def simulate(self, group_list, n_simulation = 1, probs = True):
        
        ''' simulate the entire playoff n times and also record the accumulated wins'''

        self.rounds = {}
        self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        
        for i in range(n_simulation):
            cham = self.one_time_simu(group_list)
        if probs:
            self.rounds_probs =  self._compute_probs()
            
    
    def one_time_simu(self, group_list, verbose = False, probs = False):
        
        ''' simulate the entire playoff once and also record the accumulated wins'''

        if self.team_list == None: 
            self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        round_number, done = 0, 0
        while not done: 
            all_group_winners, group_list = self.play_round(group_list)
            try:
                updated_round_stats = self.rounds[round_number]
            except KeyError:
                updated_round_stats = {}
                for team in self.team_list:
                    updated_round_stats[team] = 0
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
            if type(group_list) != list:
                done = 1
            round_number += 1
            
        return group_list

        
    def play_round(self, group_list):
        
        '''play a round of games based of a list of paired teams'''
        
        all_group_winners = [] 
        for group in group_list:
            winner = self.play_n_games(group[0], group[1])
            all_group_winners.append(winner)
        
        if len(all_group_winners) > 1:
            new_group_list = []         
            for index in range(0, len(all_group_winners), 2):
                new_group = [all_group_winners[index], all_group_winners[index + 1]]
                new_group_list.append(new_group)
                
            return all_group_winners, new_group_list
        else:  
            return all_group_winners, winner
        
        
    def play_n_games(self, team1, team2):
        
        '''simulate data, and then use our classifier to predict win/loss'''
        
        result = Game().predict(team1, team2, self.n_games_per_group)
        if sum(result[:4]) == self.winning_threshold or sum(result) >= self.winning_threshold:
            winner = team1
        else:
            winner = team2
            
        return winner
    
    
    def _compute_probs(self):
        
        '''prob = wins for a team / sum of wins for all teams at a particular round'''
        
        rounds_probs = copy.deepcopy(self.rounds)
        for round_number, round_stats in rounds_probs.items():
            m = np.sum(list(round_stats.values()))
            for k, v in rounds_probs[round_number].items():
                rounds_probs[round_number][k] = v / m
                
        return rounds_probs

# 6 Simulation
DATA = megadata.copy()

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

# initiate a playoff
playoff = FinalTournament()
# simulate the playoff 5,000 times
playoff.simulate(group_list, n_simulation = 5000)

# see the winning probabilities from 5,000 playoffs
round_df = pd.DataFrame(playoff.rounds_probs)
print(round_df)

# 7 Visualization
def plotting(rounds_data):
    
    rounds_stats = list(rounds_data.values())
    team_names = list(rounds_stats[0].keys())
    rounds_number = list(rounds_data.keys())
    states = np.array([list(r.values()) for r in rounds_stats]).T
    title_name = ['1 round' ,'2 round' ,'3 round' ,'4 round']
    
    fig = plt.figure(figsize=(16, 20))    
    for i in rounds_number :
        plt.subplot(4,1,i+1)
        x = team_names
        y = states[:,i]
        plt.bar(x, y)
        plt.xticks(x, fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Winning probabilities', fontsize = 10)
        plt.title(title_name[i], pad = 20, fontsize = 16) 
        for a,b,j in zip(x,y,range(len(x))):
            plt.text(a,b+0.001,"%.3f"%y[j],ha='center',fontsize=10)

    plt.show() 
    
    return fig

# plot the results: probabilities of winning for all teams at each round
fig = plotting(playoff.rounds_probs)

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

# reslut
print('----------------------------')
print('champion probabilities:')
print('Nets > Nuggets > Bucks')
print('----------------------------')
print('truth champion : Bucks')