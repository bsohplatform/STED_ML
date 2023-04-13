import joblib
import pandas as pd

bas_csv = pd.read_csv('BAS_DB.csv')
input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]

input_example = bas_csv[input_list].iloc[:20].to_numpy()
target_example = bas_csv['COP'].iloc[:20].to_numpy()

saved_rf = joblib.load('STED_RF.pkl')
score = saved_rf.score(input_example, target_example)
print(score)

saved_xgb = joblib.load('STED_XGB.pkl')
score = saved_xgb.score(input_example, target_example)
print(score)