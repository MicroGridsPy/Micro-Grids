#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:45:17 2019

@author: balderrama
"""


from pyDOE import lhs
lh = lhs(5, samples=60)

LLP = [0,0.05]
Diesel_Cost = [0.18,1.18]
Renewable_Invesment_Cost = [1.3,2]
Battery_Invesment_Cost = [0.4,0.7]
Generator_Invesment_Cost = [1.3,1.7]

LLP_1 = LLP[0] +lh[i,0]*(LLP[-1]-LLP[0])  
Diesel_Cost_1 = Diesel_Cost[0] +lh[i,0]*(Diesel_Cost[-1]-Diesel_Cost[0])
Renewable_Invesment_Cost_1 = Renewable_Invesment_Cost[0] +lh[i,0]*(Renewable_Invesment_Cost[-1]
                                                                    -Renewable_Invesment_Cost[0])  
Battery_Invesment_Cost_1 = Battery_Invesment_Cost[0] +lh[i,0]*(Battery_Invesment_Cost[-1]
                                                                -Battery_Invesment_Cost[0])  
Generator_Invesment_Cost_1 = Generator_Invesment_Cost[0] +lh[i,0]*(Generator_Invesment_Cost[-1]
                                                                    -Generator_Invesment_Cost[0])    