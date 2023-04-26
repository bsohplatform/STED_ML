import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sys



bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)

print(len(test_target))

'''
sys.path.append('D:\\01_Projects\\2021년 스마트플랫폼과제\\1단계\\STED_source\\Level 2')
import VCHP_layout
from HP_dataclass, ProcessFluid, Settings, Outputs


start = time.time()
for i in range(len(test_target)):
    condfluid = 'Water' if bas_data[i][0]==0.0 else 'Air'
    InCond = ProcessFluid(Y={condfluid:1.0,}, m = bas_data[i][3], T = bas_data[i][1], p = 3.0e5)
    OutCond = ProcessFluid(Y={condfluid:1.0,}, m = bas_data[i][3], T = bas_data[i][2], p = 3.0e5)
    
    evapfluid = 'Water' if bas_data[i][5]==0.0 else 'Air'
    InEvap = ProcessFluid(Y={evapfluid:1.0,}, m = 0.0, T = bas_data[i][6], p = 3.0e5)
    OutEvap = ProcessFluid(Y={evapfluid:1.0,}, m = 0.0, T = bas_data[i][7], p = 3.0e5)
    
    inputs = Settings()
    inputs.Y = {'R410A':1.0,}
    inputs.second = 'process'
    inputs.cycle = 'vcc'
    inputs.DSC = 0.01
    inputs.DSH = 5.0
    inputs.cond_dp = 0.01
    inputs.evap_dp = 0.01
    inputs.cond_type = 'phe'
    inputs.evap_type = 'phe'
    inputs.layout = 'bas'
    inputs.cond_T_pp = 1.0
    inputs.evap_T_pp = 1.0
    inputs.comp_eff = 0.74

    vchp_basic = VCHP(InCond, OutCond, InEvap, OutEvap, inputs)
    vchp_basic()

end = time.time()
print(f"플랫폼: {end - start:.5f} sec")
'''


start = time.time()
saved_rf = joblib.load('STED_RF.pkl')
score = saved_rf.score(test_input, test_target)
end = time.time()
print(f"랜덤포레스트: {end - start:.5f} sec")
print(score)

start = time.time()
saved_xgb = joblib.load('STED_XGB.pkl')
score = saved_xgb.score(test_input, test_target)
end = time.time()
print(f"XGB: {end - start:.5f} sec")
print(score)



