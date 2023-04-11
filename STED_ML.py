import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

ref_list = ['Ammonia','CO2','Ethane','Ethylene','IsoButane','Isopentane','Propylene','R11','R113','R114','R115','R116','R12','R123','R1233zd(E)','R1234yf','R1234ze(E)','R1234ze(Z)','R124','R1243zf','R125','R13','R134a','R13I1','R14','R141b','R142b','R143a','R152A','R161','R21','R218','R22','R227EA','R23','R236EA','R236FA','R245ca','R245fa','R32','R365MFC','R40','R404A','R407C','R41','R410A','R507A','RC318','Water','n-Butane','n-Pentane','n-Propane']
ref_dict = {ref_list[i]:i for i in range(len(ref_list))}

bas_csv = pd.read_csv('BAS_DB.csv')
ihx_csv = pd.read_csv('IHX_DB.csv')

fluid_h_list = []
fluid_c_list = []
ref_list = []
for fluid_h, fluid_c, ref in zip(bas_csv['fluid_h'].to_numpy(), bas_csv['fluid_c'].to_numpy(), bas_csv['Refrigerant'].to_numpy()):
    if fluid_h == 'Water':
        fluid_h_list.append(0)
    else:
        fluid_h_list.append(1)
    if fluid_c == 'Water':
        fluid_c_list.append(0)
    else:
        fluid_c_list.append(1)
    ref_list.append(ref_dict[ref])

bas_csv['fluid_h'] = fluid_h_list
bas_csv['fluid_c'] = fluid_c_list
bas_csv['Refrigerant'] = ref_list

input_list = ['fluid_h','Thi','Tho','Ph','mh','dTh','fluid_c','Tci','Tco','Pc','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]

bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv['COP'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.2, random_state=42)

rf = RandomForestRegressor(n_jobs=-1, random_state=42)
params = {'n_estimators': [50, 100, 150], 
          'max_depth':[5, 10, 15, 20],
          'min_samples_split':}
grid_cv = GridSearchCV(rf, )


