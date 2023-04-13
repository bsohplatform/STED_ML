import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor
import joblib

bas_csv = pd.read_csv('BAS_DB.csv')

input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]

bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv['COP'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)

rf = RandomForestRegressor(n_jobs=-1, n_estimators = 140, random_state=42)
params = {'max_depth': [None, 15, 10, 7]}
gs = GridSearchCV(rf, params, n_jobs=-1)
gs.fit(train_input, train_target)


print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])


#scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
#print(np.mean(scores['train_score']), np.mean(scores['test_score']))
rf.fit(train_input, train_target)

joblib.dump(rf,'STED_RF.pkl')
