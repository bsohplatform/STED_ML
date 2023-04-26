import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import joblib

'''
def xgb_eval(eta, min_child_weight, max_depth):
    params = {'eta':eta, 'min_child_weight':int(round(min_child_weight)), 'max_depth':int(round(max_depth))}
    bas_csv = pd.read_csv('BAS_DB_pre.csv')
    input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
    bas_data = bas_csv[input_list].to_numpy()
    bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
    xgb = XGBRegressor(**params, random_state=42)
    scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
    
    return np.mean(scores['test_score'])

pbounds = {'eta':(0.2, 0.7),'min_child_weight':(1, 20), 'max_depth':(1,20)}

xgb_bayesopt = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42)
xgb_bayesopt.maximize(init_point=5, n_iter=10)

print(xgb_bayesopt.max)

max_params=xgb_bayesopt.max['params']

max_params['min_child_weight'] = int(round(max_params['min_child_weight']))
max_params['max_depth'] = int(round(max_params['max_depth']))

print(max_params)


bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)

xgb_max = XGBRegressor(**max_params, random_state=42)
xgb_max.fit(train_input, train_target)

joblib.dump(xgb_max,'STED_xgb.pkl')
'''

bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
max_params = {'eta':0.219526095952917, 'min_child_weight':8, 'max_depth':14}
xgb_max = XGBRegressor(**max_params, random_state=42)
xgb_max.fit(train_input, train_target)

joblib.dump(xgb_max,'STED_xgb.pkl')
