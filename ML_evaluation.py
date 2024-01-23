import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from CoolProp.CoolProp import PropsSI
import time
import sys

ref_list = ['Ammonia','CO2','Ethane','Ethylene','IsoButane','Isopentane','Propylene','R11','R113','R114','R115','R116','R12','R123','R1233zd(E)','R1234yf','R1234ze(E)','R1234ze(Z)','R124','R1243zf','R125','R13','R134a','R13I1','R14','R141b','R142b','R143a','R152A','R161','R21','R218','R22','R227EA','R23','R236EA','R236FA','R245ca','R245fa','R32','R365MFC','R40','R404A','R407C','R41','R410A','R507A','RC318','Water','n-Butane','n-Pentane','n-Propane']
Tcrt_list = [405.4, 304.1282, 305.322, 282.35, 407.817, 460.35, 364.211, 471.06, 487.21, 418.83, 353.1, 293.03, 385.12, 456.831, 439.6, 367.85, 382.52, 423.27, 395.425, 376.93, 339.173, 301.88, 374.21, 396.44, 227.51, 477.5, 410.26, 345.857, 386.411, 375.25, 451.48, 345.02, 369.295, 374.9, 299.293, 412.44, 398.07, 447.57, 427.01, 351.255, 460.0, 416.3, 345.27, 359.345, 317.28, 344.494, 343.765, 388.38, 647.096, 425.125, 469.7, 369.89]
Tnbp_list = [239.83431861980574, 216.592, 184.56858783245625, 169.37864843190997, 261.40097716144936, 300.97633181438664, 225.5308389820667, 296.85807236462267, 320.73517445831334, 276.74149036992037, 233.93183409414397, 195.05837337845716, 243.3977126672084, 300.97304760982433, 291.41300673628854, 243.6648737793625, 254.1817060635814, 282.8777863341162, 261.187129145237, 247.72635000907877, 225.0613923699324, 191.73815855630235, 247.07616894214513,
             251.29063349165594, 145.10484437128846, 305.19535070063444, 264.0267308938689, 225.90943385012372, 249.1279497615346, 235.59668768466616, 282.0119266440489, 236.36108741137815, 232.33952525912977, 256.80907279175136, 191.13213619322852, 279.32325411141977, 271.661627105418, 298.41219978647297, 288.1983205854856, 221.49865601382044, 313.34306890893464, 249.1726857448825, 227.30321032197264, 233.02189246544975, 194.79411644275066, 221.74709913892272, 226.408980824306, 267.1753253354927, 373.1242958476844, 272.6598619089341, 309.20934582034374, 231.03621464431768]
ref_dict = {ref_list[i]:[Tcrt_list[i],Tnbp_list[i]] for i in range(len(ref_list))}

bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp']
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
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

'''
start = time.time()
saved_xgb = joblib.load('STED_XGB.pkl')
score = saved_xgb.score(test_input, test_target)
end = time.time()
print(f"XGB: {end - start:.5f} sec")
print(score)
'''

saved_xgb = joblib.load('STED_XGB.pkl')
Thi = 343.15
Tho = 353.15
Tci = 323.15
Tco = 313.15

hhi = PropsSI('H','T',Thi,'P',101300,'Water')
hho = PropsSI('H','T',Tho,'P',101300,'Water')
hci = PropsSI('H','T',Tci,'P',101300,'Water')
hco = PropsSI('H','T',Tco,'P',101300,'Water')

ref = 'R245fa'
Tcrt = ref_dict[ref][0]
Tnbp = ref_dict[ref][1]

['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp']
standard_input = [hhi,hho,Thi,Tho,2.0,0.01,hci,hco,Tci,Tco,2.0,0.01,10.0,5.0,0.75,Tcrt,Tnbp]
print(saved_xgb.predict([standard_input]))
