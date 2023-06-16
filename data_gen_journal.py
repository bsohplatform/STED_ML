from random import *
import pandas as pd
import sys, os
import CoolProp.CoolProp as CP
sys.path.append('D:\\01_Projects\\2021년 스마트플랫폼과제\\1단계\\STED_source\\Level 2')
from VCHP_layout import VCHP
from HP_dataclass import ProcessFluid, Settings, Outputs
import time


ref_list = ['R410A','R32','R290']
Tcrt_list = [CP.PropsSI('TCRIT','',0,'',0,ref) for ref in ref_list]
Tnbp_list = [CP.PropsSI('T','P',101300,'Q',0.0,ref) for ref in ref_list]
ref_dict = {ref_list[i]:[Tcrt_list[i],Tnbp_list[i]] for i in range(len(ref_list))}

Tho_ub = min(Tcrt_list)
Tho_lb = 293.15
Thi_lb = 283.15
Tci_lb = 258.15
Tco_lb = 253.15

for j in range(2):
    inputs = Settings()
    inputs.second = 'process'
    inputs.cycle = 'vcc'
    inputs.layout = 'bas' if j == 0 else 'ihx'
    inputs.cond_N_row = 3
    inputs.evap_N_row = 3
    inputs.cond_N_element = 20
    inputs.evap_N_element = 20
    inputs.expand_eff = 0.0
    inputs.mech_eff = 1.0

    if inputs.layout == 'bas':
        file_name = 'BAS.csv'
    elif inputs.layout == 'ihx':
        file_name = 'IHX.csv'
    elif inputs.layout == 'inj':
        file_name = 'INJ.csv'
        
    start = time.time()
    for i in range(150000):
        fluid_h = random()
        fluid_c = random()

        Tho = uniform(Tho_lb, Tho_ub)
        dTh = 19*random()+1
        Thi = max(Tho - dTh,Thi_lb)

        Tci = uniform(Tci_lb, Thi)
        dTc = (1.5*random()-0.5)*dTh
        Tco = max(Tci-dTc, Tco_lb)

        m_cond = 1.0


        ref_list_screen = []

        for ref, Tcrt, Tnbp in zip(ref_list, Tcrt_list, Tnbp_list):
                
            if fluid_h <= 0.5:
                Y_h = "INCOMP::MEG-50%"
                Ph = 101300
                inputs.cond_type = 'phe'
                inputs.cond_T_pp = 9.9*random()+0.1
                dTh = inputs.cond_T_pp
            else:
                Y_h = "Air"
                Ph = 101300
                inputs.cond_type = 'fthe'
                inputs.cond_T_lm = 17.0*random()+3.0
                dTh = inputs.cond_T_lm
            if fluid_c <= 0.5:
                Y_c = "INCOMP::MEG-50%"
                Pc = 101300
                inputs.evap_type = 'phe'
                inputs.evap_T_pp = 9.9*random()+0.1
                dTc = inputs.evap_T_pp
            else:
                Y_c = "Air"
                Pc = 101300
                inputs.evap_type = 'fthe'
                inputs.evap_T_lm = 17*random()+3.0
                dTc = inputs.evap_T_lm
            
            if Tcrt > 0.5*(Thi+Tho)+dTh and Tnbp < 0.5*(Tci+Tco)-dTc:
                ref_list_screen.append(ref)
            
            
        inputs.DSC = 9.9*random()+0.1
        inputs.DSH = 14.9*random()+0.1
        inputs.comp_eff = 0.6*random()+0.25
        inputs.comp_top_eff = 0.6*random()+0.25
        inputs.cond_dp = 0.1*random()+0.001
        inputs.evap_dp = 0.1*random()+0.001
        
        if inputs.layout == 'ihx':
            inputs.ihx_eff = 0.55*random()+0.4
            inputs.ihx_cold_dp = 0.1*random()+0.001
            inputs.ihx_hot_dp = 0.1*random()+0.001
        elif inputs.layout == 'inj':
            inputs.comp_top_eff = inputs.comp_eff*(1.3-0.5*random())
            
        
        results_mat = []
        for r in ref_list_screen:
            inputs.Y = {r:1.0,}
            InCond = ProcessFluid(Y={Y_h:1.0,}, m = m_cond, T = Thi, p = Ph)
            OutCond = ProcessFluid(Y={Y_h:1.0,}, m = m_cond, T = Tho, p = Ph)
            InEvap = ProcessFluid(Y={Y_c:1.0,}, m = 0.0, T = Tci, p = Pc)
            OutEvap = ProcessFluid(Y={Y_c:1.0,}, m = 0.0, T = Tco, p = Pc)
            vchp = VCHP(InCond, OutCond, InEvap, OutEvap, inputs)
            
            try:
                (InCond, OutCond, InEvap, OutEvap, InCond_REF, OutCond_REF, InEvap_REF, OutEvap_REF, outputs) = vchp()
                if InEvap.m > 0.0:                    
                    results = [outputs.COP_heating, InCond_REF.p, InCond.h, InCond.T, OutCond.h, OutCond.T, InCond.p, InCond.m, dTh, inputs.cond_dp, OutEvap_REF.p, InEvap.h, InEvap.T, OutEvap.h, OutEvap.T, InEvap.p, InEvap.m, dTc, inputs.evap_dp, outputs.DSH, inputs.DSC, inputs.comp_eff, outputs.cond_UA, outputs.evap_UA, ref_dict[r][0], ref_dict[r][1]]
                    if inputs.layout == 'ihx':
                        results.append(inputs.ihx_eff)
                        results.append(inputs.ihx_hot_dp)
                        results.append(inputs.ihx_cold_dp)
                        
                    results_mat.append(results)
            except:
                a = 0
                
        print('반복계산회수: %d'%(i))
        if not os.path.exists(file_name):    
            column_list = ['COP','Pcond','hhi','Thi','hho','Tho','Ph','mh','dTh','dPh','Pevap','hci','Tci','hco','Tco','Pc','mc','dTc','dPc','DSH','DSC','comp_eff','cond_UA','evap_UA','Tcrt','Tnbp']
            if inputs.layout == 'ihx':
                column_list.append('ihx_eff')
                column_list.append('ihx_hot_dp')
                column_list.append('ihx_cold_dp')
            df = pd.DataFrame(results_mat, columns=column_list)
            df.to_csv(file_name, index=False, mode='w', encoding='utf-8-sig')
        else:
            df = pd.DataFrame(results_mat)
            df.to_csv(file_name, index=False, mode='a', encoding='utf-8-sig', header=False)
    end = time.time()