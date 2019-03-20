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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


save = False
Village = range(80,210,12)
solar = range(10)

NPD = pd.Series()
D  = pd.Series()
for i in Village:
    sheetname = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheet_name=sheetname)    
   
    Total_Demand = Energy_Demand.sum()
    Demand = Total_Demand.sum()/len(Total_Demand)
    Discount_Rate = 0.12
    NPD.loc[i] = sum(Demand/(1+Discount_Rate)**year for year in range(1,21))
    D.loc[i] = Demand

S = pd.Series()
for s in solar:
    PV_Power =  pd.read_excel('Example/Renewable_Energy.xls', sheet_name = s)
    S.loc[s] = PV_Power[1].sum()
#    Total_PV = PV_Power[1].sum()
#   Discount_Rate = 0.12
#    NPS.loc[s] = sum(Total_PV/(1+Discount_Rate)**year for year in range(1,21))


Data = pd.read_csv('Optimization_Results_lh.csv',index_col=0)
Data = shuffle(Data, random_state=0)


for j in Data.index:
    Households = Data['Households'][j]
    PV_output = Data['PV output'][j] 
    NPD_1 = NPD[Households]
    NPS_1 = S[PV_output]
    demand = D[Households]
    solar = S[PV_output]
    Data.loc[j,'LCOE'] = (Data.loc[j,'NPC']/NPD_1)*1000
    Data.loc[j,'Demand'] = demand
    Data.loc[j,'PV energy'] = solar
    
Data.to_excel('Full_results.xls')
average = Data.mean()
Max = Data.max()
Min = Data.min()
std = Data.std()


y = Data['LCOE']
X = pd.DataFrame()
X['LLP'] = Data['LLP']*100
X['Diesel Cost'] = Data['Diesel Cost']
X['Demand'] = Data['Demand']
X['PV energy'] = Data['PV energy']
X['Renewable invesment cost'] = Data['Renewable invesment cost']
X['Genererator invesment cost'] = Data['Genererator invesment cost']
X['Battery invesment cost'] = Data['Battery invesment cost']


################# Linear regression ###########################################
lm = linear_model.LinearRegression(fit_intercept=True)
model = lm.fit(X,y)

y_predictions = lm.predict(X)
y_predictions_max = y_predictions.max()
y_predictions_nor = y_predictions/y_predictions_max

y_max = y.max()
y_nor = y/y_max



# y_true, y_pred
Score_Linear = r2_score(y,y_predictions)
MAE_linear =  mean_absolute_error(y_nor,y_predictions_nor)
RMSE_linear = mean_squared_error(y_nor,y_predictions_nor)

print('R^2 for linear regression is ' + str(Score_Linear*100) + ' %')
print('MAE for linear regression is ' + str(MAE_linear*100) + ' %')
print('RMSE for linear regression is ' + str(RMSE_linear*100) + ' %')

################ Gaussian regresion  #########################################
#kernel =1.0 * RBF(length_scale=1.0, length_scale_bounds=(0, 100.0))
#
#gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
#gp.fit(X, y)
#
#y_predictions = gp.predict(X)
#
#Score_Gaussian = round(gp.score(X,y)*100)
#
#print('R_2 for gaussian regression is ' + str(Score_Gaussian)  + ' %')
#
#
#gp_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel)
#
#
#scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
#
#scores = cross_validate(gp_1, X, y, scoring=scoring, cv=5)
#R_2 = scores['test_r2']

save = True

if save == True:
    X_1 = pd.DataFrame()
    X_1['LLP'] = Data['LLP']*100
    X_1['Diesel Cost'] = Data['Diesel Cost']
    X_1['Demand'] = Data['Demand']
    X_1['PV energy'] = Data['PV energy']
    X_1['Renewable invesment cost'] = Data['Renewable invesment cost']
    X_1['Genererator invesment cost'] = Data['Genererator invesment cost']
    X_1['Battery invesment cost'] = Data['Battery invesment cost']
    X_1['LCOE'] = Data['LCOE']
    X_1['NPC'] = Data['NPC']
    X_1.to_csv('Variables.csv')
