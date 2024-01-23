import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import CoolProp.CoolProp as CP
import time
import sys

ref_list = ['R410A','R32','R290']
Tcrt_list = [CP.PropsSI('TCRIT','',0,'',0,ref) for ref in ref_list]
Tnbp_list = [CP.PropsSI('T','P',101300,'Q',0.0,ref) for ref in ref_list]
ref_dict = {ref_list[i]:[Tcrt_list[i],Tnbp_list[i]] for i in range(len(ref_list))}


bas_csv = pd.read_csv('BAS.csv')
input_list = ['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp','Pcrt']
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap','cond_UA','evap_UA']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)


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

'''
start = time.time()
saved_rf = joblib.load('STED_RF.pkl')
score = saved_rf.score(test_input, test_target)
end = time.time()
print(f"랜덤포레스트: {end - start:.5f} sec")
print(score)
'''

start = time.time()
saved_xgb = joblib.load('STED_journal_xgb.pkl')
score = saved_xgb.score(test_input, test_target)
end = time.time()
print(f"XGB: {end - start:.5f} sec")
print(score)

saved_xgb = joblib.load('STED_journal_xgb.pkl')
Thi = 305.15
Tho = 310.15
Tci = 285.15
Tco = 280.15

flu = 'INCOMP::MEG-50%'

hhi = CP.PropsSI('H','T',Thi,'P',101300,flu)
hho = CP.PropsSI('H','T',Tho,'P',101300,flu)
hci = CP.PropsSI('H','T',Tci,'P',101300,flu)
hco = CP.PropsSI('H','T',Tco,'P',101300,flu)

ref = 'R410A'
Tcrt = ref_dict[ref][0]
Tnbp = ref_dict[ref][1]

['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp']
standard_input = [hhi,hho,Thi,Tho,2.0,0.01,hci,hco,Tci,Tco,2.0,0.01,10.0,5.0,0.75,Tcrt,Tnbp]
start = time.time()
print(saved_xgb.predict([standard_input]))
end = time.time()
print(f"XGB: {end - start:.5f} sec")