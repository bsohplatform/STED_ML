import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from xgboost import XGBRegressor
import joblib

bas_csv = pd.read_csv('BAS_DB.csv')

input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]

bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv['COP'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
xgb = XGBRegressor(tree_method = 'hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
xgb.fit(train_input, train_target)

joblib.dump(xgb,'STED_XGB.pkl')
