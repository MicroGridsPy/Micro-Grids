#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:51:54 2019

@author: balderrama
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
import matplotlib.gridspec as gridspec

i = 128
sheetname = 'village_' + str(i)
Energy_Demand = pd.read_excel('Example/Demand.xls',sheetname=sheetname)

Daily = []
iterations = int(len(Energy_Demand)/24)

for i in range(iterations):
    for j in range(1,25):
        Daily.append(j)

Energy_Demand['hour'] = Daily

Energy_Demand_Daily= Energy_Demand.groupby(['hour']).mean()/1000  

columns = []

for i in range(1,9):
    name = 'Profile ' + str(i)
    columns.append(name)

Energy_Demand_Daily.columns = columns    

for j in Energy_Demand_Daily.index:
          Energy_Demand_Daily.loc[j, 'Average'] = sum(Energy_Demand_Daily.loc[j,r] for 
                                               r in Energy_Demand_Daily.columns[:8])/8



fig, axs = plt.subplots(1, 2, figsize=(15, 10))

axs[0].plot(Energy_Demand_Daily['Profile 1'], c='b')
axs[0].plot(Energy_Demand_Daily['Profile 3'], c='r')
axs[0].plot(Energy_Demand_Daily['Profile 5'], c='y')
axs[0].plot(Energy_Demand_Daily['Profile 7'], c='m')
axs[0].set_xlim(1, 24)
axs[0].set_ylabel('Power (kW)',size =15)
axs[0].set_xlabel('Hours',size =15)
axs[0].tick_params(axis='x', which='major', labelsize=15)
axs[0].tick_params(axis='y', which='major', labelsize=15)
axs[0].text(0.4, -0.2, 'A)', transform=axs[0].transAxes, 
            size=20, weight='bold')
axs[0].legend(bbox_to_anchor=(0.8,-0.05),
    frameon=False, ncol=2,fontsize = 15)

axs[1].plot(Energy_Demand_Daily['Profile 2'], c='g')
axs[1].plot(Energy_Demand_Daily['Profile 4'], c='cyan')
axs[1].plot(Energy_Demand_Daily['Profile 6'], c='indigo')
axs[1].plot(Energy_Demand_Daily['Profile 8'], c='pink')
pylab.xlim([1,24])
axs[1].set_ylabel('Power (kW)',size =15)
axs[1].set_xlabel('Hours',size =15)
axs[1].tick_params(axis='x', which='major', labelsize=15)
axs[1].tick_params(axis='y', which='major', labelsize=15)
axs[1].text(1.6,- 0.2, 'B)', transform=axs[0].transAxes, 
            size=20, weight='bold')
axs[1].legend(bbox_to_anchor=(0.8,-0.05),
    frameon=False, ncol=2,fontsize = 15)






#
#Village = range(80,210,12)
#
#Demand_Village = pd.DataFrame()
#for i in Village:
#    sheetname = 'village_' + str(i)
#    Energy_Demand = pd.read_excel('Example/Demand.xls',sheetname=sheetname)
#    Energy_Demand['hour'] = Daily
#    Energy_Demand = Energy_Demand.groupby(['hour']).mean()/1000  
#    
#    
#    Demand_Village[sheetname] = Energy_Demand['Average']
#        
#ax = Demand_Village.plot()    
#ax.set_ylabel('Power (kW)')
#ax.set_xlabel('Hours')
#plt.legend(bbox_to_anchor=(1, 1.05))
#pylab.xlim([1,24])





