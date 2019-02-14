#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:58:37 2019

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
from matplotlib.sankey import Sankey
import plotly.plotly as py
import pylab
import enlopy as el
import matplotlib as mpl
import math as mt
import pvlib
# sena -11.4792, -67.272 tilt 12
# el remanso -12.9443,-63.9202 tilt 13
# el espino -19.3311, -63.3192 tilt 19   
# 111
Locations = ['Espino_20','Remanso_20','Sena_20']
tilt      = [19,13,12]
columns = ['output', 'direct', 'diffuse', 'temperature']
albeldo = 0.25
w=0
Solar = []
Noct = 44.8
a = (Noct-20)/800

cec_modules = pvlib.pvsystem.retrieve_sam('cecmod')
cecmodule = cec_modules.Yingli_Energy__China__YL250P_29b #select module



for loc in Locations:
    if loc == 'Espino_20':
        years = [13, 14, 15, 16, 17]
    else:
        years =  [14, 15, 16, 17]

    Solar_R = pd.DataFrame(columns=columns)
    slop = mt.radians(tilt[w])
    factor = (1-mt.cos(slop))*0.5*albeldo
    
    for i in years:
        path_1 = 'tilt/' + loc + str(i) + '.csv'
        path_2 = 'Direct/' + loc + str(i) + '_D.csv' 
        Data = pd.read_csv(path_1,index_col=0)
        Data.columns = Data.iloc[0]
        Data = Data[2:]
        Data = Data.set_index('local_time')
                
        Data_2 = pd.read_csv(path_2,index_col=0)
        Data_2.columns = Data_2.iloc[0]
        Data_2 = Data_2[2:]
        Data_2 = Data_2.set_index('local_time')
        
        for i in Data.columns:
            Data[i] =  pd.to_numeric(Data[i])
            Data_2[i] =  pd.to_numeric(Data_2[i])        
        
        
        Data_2['Total radiation'] = Data_2['direct'] + Data_2['diffuse']
        Data['reflected'] = Data_2['Total radiation']*factor
        Data['Radiation'] = Data['reflected']+Data['direct']+Data['diffuse']
        Data['Radiation'] = Data['Radiation']*1000
        Data['PV temperature'] = a*Data['Radiation'] + Data['temperature']
        
        photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
            pvlib.pvsystem.calcparams_desoto(Data['Radiation'],
                                 temp_cell=Data['PV temperature'],
                                 alpha_sc=cecmodule['alpha_sc'],
                                 a_ref = cecmodule['a_ref'],
                                 I_L_ref = cecmodule['I_L_ref'],
                                 I_o_ref = cecmodule['I_o_ref'],
                                 R_sh_ref = cecmodule['R_sh_ref'],
                                 R_s = cecmodule['R_s'],
                                 EgRef=1.121,
                                 dEgdT=-0.0002677) )
        single_diode_out = pvlib.pvsystem.singlediode(photocurrent, saturation_current,
                                  resistance_series, resistance_shunt, nNsVth)
        
        Data['PV power']= list(single_diode_out['p_mp'])
        Solar_R = Solar_R.append(Data)

    iterations = len(years)-1
    Solar_R.index = pd.to_datetime(Solar_R.index)
    for i in range(iterations):
        PV = pd.DataFrame()
        j = years[i]
        r = j+1
        
        
        
        start = '20' + str(j) + ' -03-21 01:00:00'
        end   = '20' + str(r) + ' -03-21 00:00:00'
        for s in range(1,9):
            PV[s] = Solar_R['PV power'][start:end]
        if j == 15:
            date = '20' + str(r) + '-02-29 00:00:00'
            drop = pd.date_range(date, periods=24, freq='1H')
            PV = PV.drop(drop)
        
        PV.index=range(1,8761)    
        path_3 = 'scenarios/' + loc + str(j) + '-' +str(r) + '.xls'
        PV.to_excel(path_3)
    
    w += 1
    
    
    
