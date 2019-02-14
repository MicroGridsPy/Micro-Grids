# -*- coding: utf-8 -*-
#billy rioja

from pyomo.environ import  AbstractModel

from Results import Plot_Energy_Total, Load_results1, Energy_Mix, Print_Results, Integer_Time_Series 
from Model_Creation import Model_Creation
from Model_Resolution import Model_Resolution
#21212
# Type of problem formulation:
formulation = 'LP'
#datapath='Example/Dispatch/'
# Renewable energy penetrarion

Renewable_Penetration = 0 # a number from 0 to 1.
Battery_Independency = 0  # number of days of battery independency

S = 1 # Plot scenario
Plot_Date = '25/12/2016 00:00:00' # Day-Month-Year
PlotTime = 5# Days of the plot
plot = 'No Average' # 'No Average' or 'Average'

model = AbstractModel() # define type of optimization problem

Model_Creation(model, Renewable_Penetration, Battery_Independency) # Creation of the Sets, parameters and variables.
instance = Model_Resolution(model,Renewable_Penetration,
                                Battery_Independency) # Resolution of the instance
## Upload the resulst from the instance and saving it in excel files
Data = Load_results1(instance) # Extract the results of energy from the instance and save it in a excel file 
Scenarios =  Data[3]
Scenario_Probability = Data[5].loc['Scenario Weight'] 
Generator_Data = Data[4]
Data_Renewable = Data[7]
Results = Data[2]
LCOE = Data[6]
    # Energy Plot    

Time_Series = Integer_Time_Series(instance,Scenarios, S) 
Plot_Energy_Total(instance, Time_Series, plot, Plot_Date, PlotTime)
# Data Analisys
Print_Results(instance, Generator_Data, Data_Renewable, Results, 
           LCOE,formulation)  
Energy_Mix_S = Energy_Mix(instance,Scenarios,Scenario_Probability)


