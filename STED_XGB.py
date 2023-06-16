import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import joblib

def xgb_eval(eta, min_child_weight, max_depth):
    params = {'eta':eta, 'min_child_weight':int(round(min_child_weight)), 'max_depth':int(round(max_depth))}
    bas_csv = pd.read_csv('BAS.csv')
    input_list = ['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp','Pcrt']
    bas_data = bas_csv[input_list].to_numpy()
    bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
    xgb = XGBRegressor(**params, random_state=42)
    scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
    
    return np.mean(scores['test_score'])

'''
#---------자동학습---------
pbounds = {'eta':(0.05, 0.4),'min_child_weight':(1, 20), 'max_depth':(1,20)}

xgb_bayesopt = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42)
xgb_bayesopt.maximize(init_point=10, n_iter=20)

print(xgb_bayesopt.max)

max_params=xgb_bayesopt.max['params']

max_params['min_child_weight'] = int(round(max_params['min_child_weight']))
max_params['max_depth'] = int(round(max_params['max_depth']))

print(max_params)
'''


#---------수동학습---------
max_params = {'eta': 0.07032926425886982, 'max_depth': 17, 'min_child_weight': 12}

bas_csv = pd.read_csv('BAS.csv')
input_list = ['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp','Pcrt']
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap','cond_UA','evap_UA']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
xgb_max = XGBRegressor(**max_params, random_state=42)
xgb_max.fit(train_input, train_target)

joblib.dump(xgb_max,'STED_journal_xgb.pkl')


