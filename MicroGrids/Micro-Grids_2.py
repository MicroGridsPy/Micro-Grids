# -*- coding: utf-8 -*-
#billy rioja

from pyomo.environ import  AbstractModel
import pandas as pd
import numpy as np
from Results import Plot_Energy_Total, Load_results1, Energy_Mix, Print_Results, Integer_Time_Series 
from Model_Creation import Model_Creation
from Model_Resolution import Model_Resolution
from pyomo.opt import SolverFactory
from pyDOE import lhs
import itertools

#21212
# Type of problem formulation:
formulation = 'LP'
#datapath='Example/Dispatch/'
# Renewable energy penetrarion 9.9683212008e+04
#
Renewable_Penetration = 0 # a number from 0 to 1.
Battery_Independency = 0  # number of days of battery independency
village = range(80,210,12)
Solar = range(5,10)
S = 1 # Plot scenario
Plot_Date = '25/12/2016 00:00:00' # Day-Month-Year
PlotTime = 5# Days of the plot
plot = 'No Average' # 'No Average' or 'Average'

LLP = [0,0.05]
Diesel_Cost = [0.28,1.28]
Renewable_Invesment_Cost = [1.3,1.8]
Generator_Invesment_Cost = [1.3,1.7]
Battery_Invesment_Cost = [0.4,0.6]


#lst = list(itertools.product([0, 1], repeat=5))
#lh = np.concatenate((lh,np.array(lst)))

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

Nruns = 10


for i in village:
    
    Village = 'village_' + str(i)
    Energy_Demand = pd.read_excel('Example/Demand.xls',sheetname=Village)
    
    for s in range(1,Number_Scenarios+1):
        for t in range(1, Number_Periods+1):
            instance.Energy_Demand[s,t] = Energy_Demand.iloc[t-1,s-1]
    
    
    for PV in Solar:
        lh = lhs(5, samples=Nruns)   
        Renewable_Energy = pd.read_excel('Example/Renewable_Energy.xls',sheetname=PV)        
        
        for s in range(1,Number_Scenarios+1):
            for t in range(1, Number_Periods+1):
                
                instance.Renewable_Energy_Production[s,1,t] = Renewable_Energy.iloc[t-1,s-1]        
        
        for n in range(Nruns):
            llp = LLP[0] +lh[n,0]*(LLP[-1]-LLP[0])  
            diesel_cost = Diesel_Cost[0] +lh[n,1]*(Diesel_Cost[-1]-Diesel_Cost[0])
            renewable_invesment_cost = Renewable_Invesment_Cost[0] +lh[n,2]*(Renewable_Invesment_Cost[-1]
                                                                    -Renewable_Invesment_Cost[0]) 
            generator_invesment_cost = Generator_Invesment_Cost[0] +lh[n,3]*(Generator_Invesment_Cost[-1]
                                                                    -Generator_Invesment_Cost[0]) 
            battery_invesment_cost = Battery_Invesment_Cost[0] +lh[n,4]*(Battery_Invesment_Cost[-1]
                                                                -Battery_Invesment_Cost[0])  
            
            instance.Lost_Load_Probability = round(llp,4)
                            
            Low_Heating_Value = instance.Low_Heating_Value.extract_values()[1]
            Generator_Efficiency = instance.Generator_Efficiency.extract_values()[1]
            diesel_cost = round(diesel_cost,3)
            instance.Marginal_Cost_Generator_1[1] = diesel_cost/(Low_Heating_Value*Generator_Efficiency)
            
            instance.Renewable_Invesment_Cost[1] = round(renewable_invesment_cost,3)
            
            instance.Generator_Invesment_Cost[1] = round(generator_invesment_cost,3)
            
            battery_invesment_cost = round(battery_invesment_cost,3)
            instance.Battery_Invesment_Cost = battery_invesment_cost
            
            Battery_Electronic_Invesmente_Cost = instance.Battery_Electronic_Invesmente_Cost()
            Battery_Cycles = instance.Battery_Cycles()
            Deep_of_Discharge = instance.Deep_of_Discharge()
            unitary_battery_cost = battery_invesment_cost - Battery_Electronic_Invesmente_Cost
            Unitary_Battery_Reposition_Cost = unitary_battery_cost/(Battery_Cycles*2*(1-Deep_of_Discharge))
            instance.Unitary_Battery_Reposition_Cost = Unitary_Battery_Reposition_Cost 
            
            opt = SolverFactory('cplex') # Solver use during the optimization    
            results = opt.solve(instance, tee=True) # Solving a model instance 
            instance.solutions.load_from(results)  # Loading solution into instance
            print(Village)
            print('Solar time series ' +str(PV))
            print(n)
                    
                    
            Renewable_Units = instance.Renewable_Units.get_values()[1]
                    
            Data.loc[foo, 'NPC'] = instance.ObjectiveFuntion.expr()
            Data.loc[foo, 'Households'] = i
            Data.loc[foo, 'PV output'] = PV
            Data.loc[foo, 'LLP'] = round(llp,4)
            Data.loc[foo, 'Diesel Cost'] = diesel_cost
            Data.loc[foo, 'Renewable invesment cost'] = round(renewable_invesment_cost,3) 
            Data.loc[foo, 'Genererator invesment cost'] = round(generator_invesment_cost,3)
            Data.loc[foo, 'Battery invesment cost'] = battery_invesment_cost
            Data.loc[foo, 'Battery operation cost'] = Unitary_Battery_Reposition_Cost
            Data.loc[foo, 'PV nominal capacity'] = Renewable_Nominal_Capacity*Renewable_Units
            Data.loc[foo, 'Genset nominal capacity']  =  instance.Generator_Nominal_Capacity.get_values()[1]
            Data.loc[foo, 'Battery nominal capacity'] = instance.Battery_Nominal_Capacity.get_values()[None]
            foo += 1

Data.to_excel('Results_lh_3.xls')

               
from sklearn.utils import shuffle

Data2 = shuffle(Data, random_state=0)

Data2.index = range(1,len(Data2)+1)

Data2.to_excel('Optimization_Results_lh.xls')













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
    

