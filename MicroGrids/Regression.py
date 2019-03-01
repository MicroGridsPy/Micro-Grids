#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:58:08 2019

@author: sergio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.pylab as pylab
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF


Village = range(80,210,12)
NPD = pd.Series()
for i in Village:
    sheetname = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheetname=sheetname)
    Total_Demand = Energy_Demand.sum()
    Demand = Total_Demand.sum()/len(Total_Demand)
    
    Discount_Rate = 0.12
    NPD.loc[i] = sum(Demand/(1+Discount_Rate)**year for year in range(1,21))

Data = pd.read_excel('Optimization_Results.xls',index_col=0)

for j in Data.index:
    Households = Data['Households'][j]
    NPD_1 = NPD[Households]
    Data.loc[j,'LCOE'] = (Data.loc[j,'NPC']/NPD_1)*1000
    
    
y = Data['LCOE']
X = pd.DataFrame()
X['LLP'] = Data['LLP']*100
X['Diesel Cost'] = Data['Diesel Cost']
X['Households'] = Data['Households']
X['PV output'] = Data['PV output']

lm_4 = linear_model.LinearRegression(fit_intercept=True)
model = lm_4.fit(X,y)

y_4_predictions = lm_4.predict(X)
Score_Linear = round(lm_4.score(X,y),4)
print('R_2 for linear regression is ' + str(Score_Linear*100) + ' %')

kernel =1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)

y_predictions = gp.predict(X)

Score_Gaussian = round(gp.score(X,y)*100)

print('R_2 for gaussian regression is ' + str(Score_Gaussian)  + ' %')



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(X['LLP'], X['Diesel Cost'], y, c='b')
#ax.set_xlabel('LLP (%)')
#ax.set_ylabel('Fuel Cost (USD/l)')
#ax.set_zlabel('LCOE (USD/kW)')
#
#pylab.xlim([0,6])
#pylab.ylim([0,1.2])
#ax.set_zlim(0, 0.5)
#ax.view_init(20,15)


llp = list(np.arange(0, 0.06, 0.01))
Diesel_Cost = list(np.arange(0.28, 1.48, 0.2))
Solar = range(10)

X1 = pd.DataFrame()
foo = 0
for l in llp:
    for d in Diesel_Cost:
        for v in Village:
            for s in Solar:
                X1.loc[foo,'LLP'] = l*100
                X1.loc[foo,'Diesel Cost'] = d
                X1.loc[foo,'Households'] = v
                X1.loc[foo,'PV output'] = s
                foo += 1

y_predictions_1= pd.DataFrame()                
y_predictions_1['Regression'] = gp.predict(X1)
y_predictions_1['LLP'] = X1['LLP']
y_predictions_1['Diesel Cost'] = X1['Diesel Cost']
y_predictions_1['Households'] = X1['Households']
y_predictions_1['PV output'] = X1['PV output']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1['LLP'], X1['Diesel Cost'], y_predictions_1['Regression'], c='b')
ax.set_xlabel('LLP (%)')
ax.set_ylabel('Fuel Cost (USD/l)')
ax.set_zlabel('LCOE (USD/kW)')

pylab.xlim([0,6])
pylab.ylim([0,1.2])
ax.set_zlim(0, 0.5)
ax.view_init(20,45)





