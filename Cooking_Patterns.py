#%% Definition of the inputs
'''
Input data definition (this is planned to be externalised in a separate script)
'''

#Create new user classes
HI = User("high income",11,3)
User_list.append(HI)

HMI = User("higher middle income",38,3)
User_list.append(HMI)

LMI = User("lower middle income",34,3)
User_list.append(LMI)

LI = User("low income",45,3)
User_list.append(LI)


#Create new appliances

#High-Income
HI_indoor_bulb = HI.Appliance(HI,6,7,2,120,0.2,10)
HI_indoor_bulb.windows([1170,1440],[0,30],0.35)

HI_outdoor_bulb = HI.Appliance(HI,2,13,2,600,0.2,10)
HI_outdoor_bulb.windows([0,330],[1170,1440],0.35)

HI_TV = HI.Appliance(HI,2,60,3,180,0.1,5)
HI_TV.windows([720,900],[1170,1440],0.35,[0,60])

HI_DVD = HI.Appliance(HI,1,8,3,60,0.1,5)
HI_DVD.windows([720,900],[1170,1440],0.35,[0,60])

HI_Antenna = HI.Appliance(HI,1,8,3,120,0.1,5)
HI_Antenna.windows([720,900],[1170,1440],0.35,[0,60])

HI_Phone_charger = HI.Appliance(HI,5,2,2,300,0.2,5)
HI_Phone_charger.windows([1110,1440],[0,30],0.35)

HI_Freezer = HI.Appliance(HI,1,200,1,1440,0,30,'yes',3)
HI_Freezer.windows([0,1440],[0,0])
HI_Freezer.specific_cycle_1(200,20,5,10)
HI_Freezer.specific_cycle_2(200,15,5,15)
HI_Freezer.specific_cycle_3(200,10,5,20)
HI_Freezer.cycle_behaviour([480,1200],[0,0],[300,479],[0,0],[0,299],[1201,1440])

HI_Freezer2 = HI.Appliance(HI,1,200,1,1440,0,30,'yes',3)
HI_Freezer2.windows([0,1440],[0,0])
HI_Freezer2.specific_cycle_1(200,20,5,10)
HI_Freezer2.specific_cycle_2(200,15,5,15)
HI_Freezer2.specific_cycle_3(200,10,5,20)
HI_Freezer2.cycle_behaviour([480,1200],[0,0],[300,479],[0,0],[0,299],[1201,1440])

HI_Mixer = HI.Appliance(HI,1,50,3,30,0.1,1,occasional_use = 0.33)
HI_Mixer.windows([420,480],[660,750],0.35,[1140,1200])

#Higher-Middle Income
HMI_indoor_bulb = HMI.Appliance(HMI,5,7,2,120,0.2,10)
HMI_indoor_bulb.windows([1170,1440],[0,30],0.35)

HMI_outdoor_bulb = HMI.Appliance(HMI,2,13,2,600,0.2,10)
HMI_outdoor_bulb.windows([0,330],[1170,1440],0.35)

HMI_TV = HMI.Appliance(HMI,1,60,2,120,0.1,5)
HMI_TV.windows([1170,1440],[0,60],0.35)

HMI_DVD = HMI.Appliance(HMI,1,8,2,40,0.1,5)
HMI_DVD.windows([1170,1440],[0,60],0.35)

HMI_Antenna = HMI.Appliance(HMI,1,8,2,80,0.1,5)
HMI_Antenna.windows([1170,1440],[0,60],0.35)

HMI_Radio = HMI.Appliance(HMI,1,36,2,60,0.1,5)
HMI_Radio.windows([390,450],[1140,1260],0.35)

HMI_Phone_charger = HMI.Appliance(HMI,4,2,2,300,0.2,5)
HMI_Phone_charger.windows([1110,1440],[0,30],0.35)

HMI_Freezer = HMI.Appliance(HMI,1,200,1,1440,0,30, 'yes',3)
HMI_Freezer.windows([0,1440],[0,0])
HMI_Freezer.specific_cycle_1(200,20,5,10)
HMI_Freezer.specific_cycle_2(200,15,5,15)
HMI_Freezer.specific_cycle_3(200,10,5,20)
HMI_Freezer.cycle_behaviour([480,1200],[0,0],[300,479],[0,0],[0,299],[1201,1440])

HMI_Mixer = HMI.Appliance(HMI,1,50,3,30,0.1,1, occasional_use = 0.33)
HMI_Mixer.windows([420,450],[660,750],0.35,[1020,1170])

#Lower-Midlle Income
LMI_indoor_bulb = LMI.Appliance(LMI,3,7,2,120,0.2,10)
LMI_indoor_bulb.windows([1170,1440],[0,30],0.35)

LMI_outdoor_bulb = LMI.Appliance(LMI,2,13,2,600,0.2,10)
LMI_outdoor_bulb.windows([0,330],[1170,1440],0.35)

LMI_TV = LMI.Appliance(LMI,1,60,3,90,0.1,5)
LMI_TV.windows([450,660],[720,840],0.35,[1170,1440])

LMI_DVD = LMI.Appliance(LMI,1,8,3,30,0.1,5)
LMI_DVD.windows([450,660],[720,840],0.35,[1170,1440])

LMI_Antenna = LMI.Appliance(LMI,1,8,3,60,0.1,5)
LMI_Antenna.windows([450,660],[720,840],0.35,[1170,1440])

LMI_Phone_charger = LMI.Appliance(LMI,4,2,1,300,0.2,5)
LMI_Phone_charger.windows([1020,1440],[0,0],0.35)

LMI_Mixer = LMI.Appliance(LMI,1,50,2,30,0.1,1, occasional_use = 0.33)
LMI_Mixer.windows([660,750],[1110,1200],0.35)

#Low Income
LI_indoor_bulb = LI.Appliance(LI,2,7,2,120,0.2,10)
LI_indoor_bulb.windows([1170,1440],[0,30],0.35)

LI_outdoor_bulb = LI.Appliance(LI,1,13,2,600,0.2,10)
LI_outdoor_bulb.windows([0,330],[1170,1440],0.35)

LI_TV = LI.Appliance(LI,1,60,3,90,0.1,5)
LI_TV.windows([750,840],[1170,1440],0.35,[0,30])

LI_DVD = LI.Appliance(LI,1,8,3,30,0.1,5)
LI_DVD.windows([750,840],[1170,1440],0.35,[0,30])

LI_Antenna = LI.Appliance(LI,1,8,3,60,0.1,5)
LI_Antenna.windows([750,840],[1170,1440],0.35,[0,30])

LI_Phone_charger = LI.Appliance(LI,2,2,1,300,0.2,5)
LI_Phone_charger.windows([1080,1440],[0,0],0.35)


#Create Cooking appliances

#High-Income
HI_lunch1_soup = HI.Appliance(HI,1,1800,2,70,0.15,60, thermal_P_var = 0.2, pref_index =1, fixed_cycle=1)
HI_lunch1_soup.windows([12*60,15*60],[0,0],0.15)
HI_lunch1_soup.specific_cycle_1(1800,10,750,60,0.15)
HI_lunch1_soup.cycle_behaviour([12*60,15*60],[0,0])

HI_lunch2_rice = HI.Appliance(HI,1,1800,2,25,0.15,20, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
HI_lunch2_rice.windows([12*60,15*60],[0,0],0.15)
HI_lunch2_rice.specific_cycle_1(1800,10,750,15,0.15)
HI_lunch2_rice.cycle_behaviour([12*60,15*60],[0,0])

HI_lunch2_egg = HI.Appliance(HI,1,1200,2,3,0.2,3, thermal_P_var = 0.2 , pref_index = 2)
HI_lunch2_egg.windows([12*60,15*60],[0,0],0.15)

HI_lunch2_platano = HI.Appliance(HI,1,1800,2,10,0.15,5, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
HI_lunch2_platano.windows([12*60,15*60],[0,0],0.15)
HI_lunch2_platano.specific_cycle_1(1800,5,1200,5,0.15)
HI_lunch2_platano.cycle_behaviour([12*60,15*60],[0,0])

HI_lunch2_meat = HI.Appliance(HI,1,1200,2,7,0.15,3, thermal_P_var = 0.2, pref_index = 2)
HI_lunch2_meat.windows([12*60,15*60],[0,0],0.15)

HI_lunch3_beansnrice = HI.Appliance(HI,1,1800,2,45,0.2,30, thermal_P_var =0.2 , pref_index = 3, fixed_cycle=1)
HI_lunch3_beansnrice.windows([12*60,15*60],[0,0],0.15)
HI_lunch3_beansnrice.specific_cycle_1(1800,10,750,35,0.2)
HI_lunch3_beansnrice.cycle_behaviour([12*60,15*60],[0,0])

HI_lunch3_meat = HI.Appliance(HI,1,1200,2,10,0.2,5, thermal_P_var = 0.2, pref_index = 3)
HI_lunch3_meat.windows([12*60,15*60],[0,0],0.15)

HI_lunch_yuca = HI.Appliance(HI,1,1800,1,25,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HI_lunch_yuca.windows([13*60,14*60],[0,0],0.15)
HI_lunch_yuca.specific_cycle_1(1800,10,750,15,0.15)
HI_lunch_yuca.cycle_behaviour([12*60,15*60],[0,0])

HI_breakfast_huminta = HI.Appliance(HI,1,1800,1,65,0.15,50, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HI_breakfast_huminta.windows([6*60,9*60],[0,0],0.15)
HI_breakfast_huminta.specific_cycle_1(1800,5,750,60,0.15)
HI_breakfast_huminta.cycle_behaviour([6*60,9*60],[0,0])

HI_breakfast_bread = HI.Appliance(HI,1,1800,1,15,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HI_breakfast_bread.windows([6*60,9*60],[0,0],0.15)
HI_breakfast_bread.specific_cycle_1(1800,10,1200,5,0.15)
HI_breakfast_bread.cycle_behaviour([6*60,9*60],[0,0])

HI_breakfast_coffee = HI.Appliance(HI,1,1800,1,5,0.15,2, thermal_P_var = 0.2, pref_index =0)
HI_breakfast_coffee.windows([6*60,9*60],[0,0],0.15)

HI_mate = HI.Appliance(HI,1,1800,1,30,0.3,2, thermal_P_var = 0.2, pref_index =0)
HI_mate.windows([7*60,20*60],[0,0],0.15)

#HighMedium-Income

HMI_lunch1_soup = HMI.Appliance(HMI,1,1800,2,70,0.15,60, thermal_P_var = 0.2, pref_index =1, fixed_cycle=1)
HMI_lunch1_soup.windows([12*60,15*60],[0,0],0.15)
HMI_lunch1_soup.specific_cycle_1(1800,10,750,60,0.15)
HMI_lunch1_soup.cycle_behaviour([12*60,15*60],[0,0])

HMI_lunch2_rice = HMI.Appliance(HMI,1,1800,2,25,0.15,20, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
HMI_lunch2_rice.windows([12*60,15*60],[0,0],0.15)
HMI_lunch2_rice.specific_cycle_1(1800,10,750,15,0.15)
HMI_lunch2_rice.cycle_behaviour([12*60,15*60],[0,0])

HMI_lunch2_egg = HMI.Appliance(HMI,1,1200,2,3,0.2,3, thermal_P_var = 0.2 , pref_index = 2)
HMI_lunch2_egg.windows([12*60,15*60],[0,0],0.15)

HMI_lunch2_platano = HMI.Appliance(HMI,1,1800,2,10,0.15,5, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
HMI_lunch2_platano.windows([12*60,15*60],[0,0],0.15)
HMI_lunch2_platano.specific_cycle_1(1800,5,1200,5,0.15)
HMI_lunch2_platano.cycle_behaviour([12*60,15*60],[0,0])

HMI_lunch2_meat = HMI.Appliance(HMI,1,1200,2,7,0.15,3, thermal_P_var = 0.2, pref_index = 2)
HMI_lunch2_meat.windows([12*60,15*60],[0,0],0.15)

HMI_lunch3_beansnrice = HMI.Appliance(HMI,1,1800,2,45,0.2,30, thermal_P_var =0.2 , pref_index = 3, fixed_cycle=1)
HMI_lunch3_beansnrice.windows([12*60,15*60],[0,0],0.15)
HMI_lunch3_beansnrice.specific_cycle_1(1800,10,750,35,0.2)
HMI_lunch3_beansnrice.cycle_behaviour([12*60,15*60],[0,0])

HMI_lunch3_meat = HMI.Appliance(HMI,1,1200,2,10,0.2,5, thermal_P_var = 0.2, pref_index = 3)
HMI_lunch3_meat.windows([12*60,15*60],[0,0],0.15)

HMI_lunch_yuca = HMI.Appliance(HMI,1,1800,1,25,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HMI_lunch_yuca.windows([13*60,14*60],[0,0],0.15)
HMI_lunch_yuca.specific_cycle_1(1800,10,750,15,0.15)
HMI_lunch_yuca.cycle_behaviour([12*60,15*60],[0,0])

HMI_breakfast_huminta = HMI.Appliance(HMI,1,1800,1,65,0.15,50, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HMI_breakfast_huminta.windows([6*60,9*60],[0,0],0.15)
HMI_breakfast_huminta.specific_cycle_1(1800,5,750,60,0.15)
HMI_breakfast_huminta.cycle_behaviour([6*60,9*60],[0,0])

HMI_breakfast_bread = HMI.Appliance(HMI,1,1800,1,15,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
HMI_breakfast_bread.windows([6*60,9*60],[0,0],0.15)
HMI_breakfast_bread.specific_cycle_1(1800,10,1200,5,0.15)
HMI_breakfast_bread.cycle_behaviour([6*60,9*60],[0,0])

HMI_breakfast_coffee = HMI.Appliance(HMI,1,1800,1,5,0.15,2, thermal_P_var = 0.2, pref_index =0)
HMI_breakfast_coffee.windows([6*60,9*60],[0,0],0.15)

HMI_mate = HMI.Appliance(HMI,1,1800,1,30,0.3,2, thermal_P_var = 0.2, pref_index =0)
HMI_mate.windows([7*60,20*60],[0,0],0.15)

#LowMedium-Income

LMI_lunch1_soup = LMI.Appliance(LMI,1,1800,2,70,0.15,60, thermal_P_var = 0.2, pref_index =1, fixed_cycle=1)
LMI_lunch1_soup.windows([12*60,15*60],[0,0],0.15)
LMI_lunch1_soup.specific_cycle_1(1800,10,750,60,0.15)
LMI_lunch1_soup.cycle_behaviour([12*60,15*60],[0,0])

LMI_lunch2_rice = LMI.Appliance(LMI,1,1800,2,25,0.15,20, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
LMI_lunch2_rice.windows([12*60,15*60],[0,0],0.15)
LMI_lunch2_rice.specific_cycle_1(1800,10,750,15,0.15)
LMI_lunch2_rice.cycle_behaviour([12*60,15*60],[0,0])

LMI_lunch2_egg = LMI.Appliance(LMI,1,1200,2,3,0.2,3, thermal_P_var = 0.2 , pref_index = 2)
LMI_lunch2_egg.windows([12*60,15*60],[0,0],0.15)

LMI_lunch2_platano = LMI.Appliance(LMI,1,1800,2,10,0.15,5, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
LMI_lunch2_platano.windows([12*60,15*60],[0,0],0.15)
LMI_lunch2_platano.specific_cycle_1(1800,5,1200,5,0.15)
LMI_lunch2_platano.cycle_behaviour([12*60,15*60],[0,0])

LMI_lunch2_meat = LMI.Appliance(LMI,1,1200,2,7,0.15,3, thermal_P_var = 0.2, pref_index = 2)
LMI_lunch2_meat.windows([12*60,15*60],[0,0],0.15)

LMI_lunch3_beansnrice = LMI.Appliance(LMI,1,1800,2,45,0.2,30, thermal_P_var =0.2 , pref_index = 3, fixed_cycle=1)
LMI_lunch3_beansnrice.windows([12*60,15*60],[0,0],0.15)
LMI_lunch3_beansnrice.specific_cycle_1(1800,10,750,35,0.2)
LMI_lunch3_beansnrice.cycle_behaviour([12*60,15*60],[0,0])

LMI_lunch3_meat = LMI.Appliance(LMI,1,1200,2,10,0.2,5, thermal_P_var = 0.2, pref_index = 3)
LMI_lunch3_meat.windows([12*60,15*60],[0,0],0.15)

LMI_lunch_yuca = LMI.Appliance(LMI,1,1800,1,25,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LMI_lunch_yuca.windows([13*60,14*60],[0,0],0.15)
LMI_lunch_yuca.specific_cycle_1(1800,10,750,15,0.15)
LMI_lunch_yuca.cycle_behaviour([12*60,15*60],[0,0])

LMI_breakfast_huminta = LMI.Appliance(LMI,1,1800,1,65,0.15,50, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LMI_breakfast_huminta.windows([6*60,9*60],[0,0],0.15)
LMI_breakfast_huminta.specific_cycle_1(1800,5,750,60,0.15)
LMI_breakfast_huminta.cycle_behaviour([6*60,9*60],[0,0])

LMI_breakfast_bread = LMI.Appliance(LMI,1,1800,1,15,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LMI_breakfast_bread.windows([6*60,9*60],[0,0],0.15)
LMI_breakfast_bread.specific_cycle_1(1800,10,1200,5,0.15)
LMI_breakfast_bread.cycle_behaviour([6*60,9*60],[0,0])

LMI_breakfast_coffee = LMI.Appliance(LMI,1,1800,1,5,0.15,2, thermal_P_var = 0.2, pref_index =0)
LMI_breakfast_coffee.windows([6*60,9*60],[0,0],0.15)

LMI_mate = LMI.Appliance(LMI,1,1800,1,30,0.3,2, thermal_P_var = 0.2, pref_index =0)
LMI_mate.windows([7*60,20*60],[0,0],0.15)

#Low-Income

LI_lunch1_soup = LI.Appliance(LI,1,1800,2,70,0.15,60, thermal_P_var = 0.2, pref_index =1, fixed_cycle=1)
LI_lunch1_soup.windows([12*60,15*60],[0,0],0.15)
LI_lunch1_soup.specific_cycle_1(1800,10,750,60,0.15)
LI_lunch1_soup.cycle_behaviour([12*60,15*60],[0,0])

LI_lunch2_rice = LI.Appliance(LI,1,1800,2,25,0.15,20, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
LI_lunch2_rice.windows([12*60,15*60],[0,0],0.15)
LI_lunch2_rice.specific_cycle_1(1800,10,750,15,0.15)
LI_lunch2_rice.cycle_behaviour([12*60,15*60],[0,0])

LI_lunch2_egg = LI.Appliance(LI,1,1200,2,3,0.2,3, thermal_P_var = 0.2 , pref_index = 2)
LI_lunch2_egg.windows([12*60,15*60],[0,0],0.15)

LI_lunch2_platano = LI.Appliance(LI,1,1800,2,10,0.15,5, thermal_P_var = 0.2, pref_index = 2, fixed_cycle=1)
LI_lunch2_platano.windows([12*60,15*60],[0,0],0.15)
LI_lunch2_platano.specific_cycle_1(1800,5,1200,5,0.15)
LI_lunch2_platano.cycle_behaviour([12*60,15*60],[0,0])

LI_lunch2_meat = LI.Appliance(LI,1,1200,2,7,0.15,3, thermal_P_var = 0.2, pref_index = 2)
LI_lunch2_meat.windows([12*60,15*60],[0,0],0.15)

LI_lunch3_beansnrice = LI.Appliance(LI,1,1800,2,45,0.2,30, thermal_P_var =0.2 , pref_index = 3, fixed_cycle=1)
LI_lunch3_beansnrice.windows([12*60,15*60],[0,0],0.15)
LI_lunch3_beansnrice.specific_cycle_1(1800,10,750,35,0.2)
LI_lunch3_beansnrice.cycle_behaviour([12*60,15*60],[0,0])

LI_lunch3_meat = LI.Appliance(LI,1,1200,2,10,0.2,5, thermal_P_var = 0.2, pref_index = 3)
LI_lunch3_meat.windows([12*60,15*60],[0,0],0.15)

LI_lunch_yuca = LI.Appliance(LI,1,1800,1,25,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LI_lunch_yuca.windows([13*60,14*60],[0,0],0.15)
LI_lunch_yuca.specific_cycle_1(1800,10,750,15,0.15)
LI_lunch_yuca.cycle_behaviour([12*60,15*60],[0,0])

LI_breakfast_huminta = LI.Appliance(LI,1,1800,1,65,0.15,50, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LI_breakfast_huminta.windows([6*60,9*60],[0,0],0.15)
LI_breakfast_huminta.specific_cycle_1(1800,5,750,60,0.15)
LI_breakfast_huminta.cycle_behaviour([6*60,9*60],[0,0])

LI_breakfast_bread = LI.Appliance(LI,1,1800,1,15,0.15,10, thermal_P_var = 0.2, pref_index =0, fixed_cycle=1)
LI_breakfast_bread.windows([6*60,9*60],[0,0],0.15)
LI_breakfast_bread.specific_cycle_1(1800,10,1200,5,0.15)
LI_breakfast_bread.cycle_behaviour([6*60,9*60],[0,0])

LI_breakfast_coffee = LI.Appliance(LI,1,1800,1,5,0.15,2, thermal_P_var = 0.2, pref_index =0)
LI_breakfast_coffee.windows([6*60,9*60],[0,0],0.15)

LI_mate = LI.Appliance(LI,1,1800,1,30,0.3,2, thermal_P_var = 0.2, pref_index =0)
LI_mate.windows([7*60,20*60],[0,0],0.15)