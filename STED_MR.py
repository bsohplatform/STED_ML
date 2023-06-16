import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

bas_csv = pd.read_csv('BAS_DB_pre.csv')
input_list = ['hhi','hho','Thi','Tho','dTh','dPh','hci','hco','Tci','Tco','dTc','dPc','DSH','DSC','comp_eff','Tcrt','Tnbp']
bas_data = bas_csv[input_list].to_numpy()
bas_target = bas_csv[['COP','Pcond','Pevap']].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)

ss = StandardScaler()
train_input_scaled = ss.fit_transform(train_input)
test_input_scaled = ss.fit_transform(test_input)


train_score_array = []
test_score_array = []
for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree,include_bias=False)
    poly.fit(train_input_scaled)
    train_input_scaled_poly = poly.transform(train_input_scaled)
    test_input_scaled_poly = poly.transform(test_input_scaled)
    lr = LinearRegression()
    lr.fit(train_input_scaled_poly, train_target)
    rid = Ridge()
    rid.fit(train_input_scaled_poly, train_target)
    las = Lasso()
    las.fit(train_input_scaled_poly, train_target)
    train_score_array.append([lr.score(train_input_scaled_poly, train_target), rid.score(train_input_scaled_poly, train_target), las.score(train_input_scaled_poly, train_target)])
    test_score_array.append([lr.score(test_input_scaled_poly, test_target), rid.score(test_input_scaled_poly, test_target), las.score(test_input_scaled_poly, test_target)])
    

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(5), train_score_array[:][0],'ko-', label='LR-train')
ax.plot(range(5), test_score_array[:][0],'ko--', label='LR-test')
ax.plot(range(5), train_score_array[:][0],'ro-', label='RD-train')
ax.plot(range(5), test_score_array[:][0],'ro--', label='RD-test')
ax.plot(range(5), train_score_array[:][0],'bo-', label='LS-train')
ax.plot(range(5), test_score_array[:][0],'bo--', label='LS-test')

ax.legend()

plt.savefig('Multivariate Regression.png')