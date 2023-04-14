import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import joblib

def xgb_eval(eta, min_child_weight, max_depth, subsample, colsample_bytree):
    params = {'eta':eta, 'min_child_weight':int(round(min_child_weight)), 'max_depth':int(round(max_depth)), 'subsample':subsample, 'colsample_bytree':colsample_bytree}
    bas_csv = pd.read_csv('BAS_DB.csv')
    input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
    bas_data = bas_csv[input_list].to_numpy()
    bas_target = bas_csv['COP'].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
    xgb = XGBRegressor(**params, random_state=42)
    scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
    
    return np.mean(scores['test_score'])

pbounds = {'eta':(0.05, 0.4),'min_child_weight':(1, 20), 'max_depth':(1,20), 'subsample':(0.5, 0.1),'colsample_bytree':(0.1, 0.9)}

joblib.dump(xgb_eval,'STED_XGB.pkl')

xgb_bayesopt = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42)
xgb_bayesopt.maximize(init_point=5, n_iter=10)

print(xgb_bayesopt.max)

max_params=xgb_bayesopt.max['params']

max_params['min_child_weight'] = int(round(max_params['min_child_weight']))
max_params['max_depth'] = int(round(max_params['max_depth']))

print(max_params)


bas_csv = pd.read_csv('BAS_DB.csv')
input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv['COP'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
xgb_max = XGBRegressor(**max_params, random_state=42)
xgb_max.fit(train_input, train_target)

joblib.dump(xgb_max,'STED_xgb.pkl')