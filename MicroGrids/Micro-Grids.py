# -*- coding: utf-8 -*-
#billy rioja

from pyomo.environ import  AbstractModel
import pandas as pd
import numpy as np
from Results import Plot_Energy_Total, Load_results1, Energy_Mix, Print_Results, Integer_Time_Series 
from Model_Creation import Model_Creation
from Model_Resolution import Model_Resolution
from pyomo.opt import SolverFactory
#21212
# Type of problem formulation:
formulation = 'LP'
#datapath='Example/Dispatch/'
# Renewable energy penetrarion 9.9683212008e+04
#
Renewable_Penetration = 0 # a number from 0 to 1.
Battery_Independency = 0  # number of days of battery independency
village = range(80,210,12)
Solar = range(10)
S = 1 # Plot scenario
Plot_Date = '25/12/2016 00:00:00' # Day-Month-Year
PlotTime = 5# Days of the plot
plot = 'No Average' # 'No Average' or 'Average'
LLP = [0.02]
Diesel_Cost = list(np.arange(0.18, 1.38, 0.2))
 # define type of optimization problem
model = AbstractModel()

Model_Creation(model, Renewable_Penetration, Battery_Independency)
instance = Model_Resolution(model, Renewable_Penetration, Battery_Independency) 
Number_Scenarios = int(instance.Scenarios.extract_values()[None])
Number_Periods = int(instance.Periods.extract_values()[None])
Number_Renewable_Source = int(instance.Renewable_Source.extract_values()[None])
foo = 0
Data = pd.DataFrame()
Results = pd.DataFrame()
Renewable_Nominal_Capacity = instance.Renewable_Nominal_Capacity.extract_values()[1]

for i in village:
       
    Village = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheetname=Village)
    
    for s in range(1,Number_Scenarios+1):
        for t in range(1, Number_Periods+1):
            instance.Energy_Demand[s,t] = Energy_Demand.iloc[t-1,s-1]
    
    
    for PV in Solar:
        Renewable_Energy = pd.read_excel('Example/Renewable_Energy.xls',sheetname=PV)        
        
        for s in range(1,Number_Scenarios+1):
            for t in range(1, Number_Periods+1):
                
                instance.Renewable_Energy_Production[s,1,t] = Renewable_Energy.iloc[t-1,s-1]        
        
        for llp in LLP:
            instance.Lost_Load_Probability = llp
            
            
           
            
            for diesel_cost in Diesel_Cost:
                
                Low_Heating_Value = instance.Low_Heating_Value.extract_values()[1]
                Generator_Efficiency = instance.Generator_Efficiency.extract_values()[1]
                
                instance.Marginal_Cost_Generator_1[1] = diesel_cost/(Low_Heating_Value*Generator_Efficiency)

                
                opt = SolverFactory('cplex') # Solver use during the optimization    
                results = opt.solve(instance, tee=True) # Solving a model instance 
                instance.solutions.load_from(results)  # Loading solution into instance
                print(Village)
                print('Solar time series ' +str(PV))
                print('Lost load probability ' + str(llp*100) + ' %')
                print('Diesel cost ' + str(diesel_cost) + ' USD/l')    
                
                
                Renewable_Units = instance.Renewable_Units.get_values()[1]
                
                Data.loc[foo, 'NPC'] = instance.ObjectiveFuntion.expr()
                Data.loc[foo, 'Households'] = i
                Data.loc[foo, 'PV output'] = PV
                Data.loc[foo, 'LLP'] = llp
                Data.loc[foo, 'Diesel Cost'] = diesel_cost
                Data.loc[foo, 'PV nominal capacity'] = Renewable_Nominal_Capacity*Renewable_Units
                Data.loc[foo, 'Genset nominal capacity']  =  instance.Generator_Nominal_Capacity.get_values()[1]
                Data.loc[foo, 'Battery nominal capacity'] = instance.Battery_Nominal_Capacity.get_values()[None]
                foo += 1


Data.to_excel('Results1.xls')


                




















    ## Upload the resulst from the instance and saving it in excel files
#    Data = Load_results1(instance) # Extract the results of energy from the instance and save it in a excel file 
#    Scenarios =  Data[3]
#    Scenario_Probability = Data[5].loc['Scenario Weight'] 
#    Generator_Data = Data[4]
#    Data_Renewable = Data[7]
#    Results = Data[2]
#    LCOE = Data[6]
#        # Energy Plot    
#    
#    Time_Series = Integer_Time_Series(instance,Scenarios, S) 
#    Plot_Energy_Total(instance, Time_Series, plot, Plot_Date, PlotTime)
#    # Data Analisys
#    Print_Results(instance, Generator_Data, Data_Renewable, Results, 
#               LCOE,formulation)  
#    Energy_Mix_S = Energy_Mix(instance,Scenarios,Scenario_Probability)
    

