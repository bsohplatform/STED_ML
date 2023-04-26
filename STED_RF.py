import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
import joblib
'''
# Bayesian Optimization

def rf_eval(min_samples_split, min_samples_leaf):
    params = {'n_estimators':100,'min_samples_split':int(round(min_samples_split)), 'min_samples_leaf':int(round(min_samples_leaf))}
    bas_csv = pd.read_csv('BAS_DB_pre.csv')
    input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant']
    bas_data = bas_csv[input_list].to_numpy()
    bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
    rf = RandomForestRegressor(**params, random_state=42,n_jobs=-1)
    scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
    
    return np.mean(scores['test_score'])

pbounds = {'min_samples_split':(2, 20), 'min_samples_leaf':(1, 20)}
rf_bayesopt = BayesianOptimization(f=rf_eval, pbounds=pbounds, random_state=42)
rf_bayesopt.maximize(init_point=5, n_iter=10)

print(rf_bayesopt.max)

max_params=rf_bayesopt.max['params']

max_params['min_samples_split'] = int(round(max_params['min_samples_split']))
max_params['min_samples_leaf'] = int(round(max_params['min_samples_leaf']))

print(max_params)


bas_csv = pd.read_csv('BAS_DB.csv')
input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv['COP'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
rf_max = RandomForestRegressor(**max_params, random_state=42, n_jobs=-1)
rf_max.fit(train_input, train_target)

joblib.dump(rf_max,'STED_RF.pkl')
'''

# Save with Pickle form
bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
rf = RandomForestRegressor(min_samples_split=2,min_samples_leaf=1,random_state=42,n_jobs=-1)

scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['test_score']))

rf.fit(train_input, train_target)

joblib.dump(rf, 'STED_RF.pkl')
