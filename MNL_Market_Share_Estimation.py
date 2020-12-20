
import biogeme.database as db
import pandas as pd
import numpy as np
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, DefineVariable, bioMultSum
import sys
import biogeme.results as res





# Read the data

df = pd.read_csv('Results_All_Merged_Work.csv', sep = ';' )
pd.set_option('display.max_row', 23000)
pd.set_option('display.max_columns', 250)
pd.options.mode.chained_assignment = None
final_df = df[['q0003','q0004','q0005','q0006','q0008','q0009','q0036','q0007']]



final_df['q0008']=final_df['q0008'].str.strip().replace('',np.nan)
final_df['q0007']=final_df['q0007'].str.strip().replace('',np.nan)
final_df['q0006']=final_df['q0006'].str.strip().replace('',np.nan)
final_df['q0036']=final_df['q0036'].str.strip().replace('',np.nan)
final_df['q0009']=final_df['q0009'].str.strip().replace('',np.nan)

final_df.dropna( inplace=True)

final_df['q0008']=final_df['q0008'].astype(str).astype(int)
final_df['q0007']=final_df['q0007'].astype(str).astype(int)
final_df['q0006']=final_df['q0006'].astype(str).astype(int)
final_df['q0036']=final_df['q0036'].astype(str).astype(int)
final_df['q0009']=final_df['q0009'].astype(str).astype(int)

final_df = final_df.rename(columns={'q0007': 'CHOICE'})
final_df = final_df.rename(columns={'q0003': 'Age'})
final_df = final_df.rename(columns={'q0004': 'Gender'})
final_df = final_df.rename(columns={'q0005': 'Job_Status'})
final_df = final_df.rename(columns={'q0006': 'Workplace'})
final_df = final_df.rename(columns={'q0008': 'Frequency'})
final_df = final_df.rename(columns={'q0009': 'Duration'})
final_df = final_df.rename(columns={'q0036': 'Household_Type'})

normalizedWeight = np.ones(len(final_df['CHOICE']),dtype=int)
final_df['Weight'] = normalizedWeight

print(normalizedWeight)

database = db.Database('MNL_model', final_df  )

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)






#Removing some observations
"""exclude = (q0005 == 9) * (q0005 == 10)
for q in range(nbrQuestions):
    exclude = exclude + (Variable(f'CHOICE_{q}') == 15)
database.remove(exclude > 0)"""

print(f'The database has {database.data.shape[0]} observations, '
      f'and {database.data.shape[1]} columns')



final_df['CHOICE'].replace(    to_replace=[1,2,3,4 ],    value=1,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[5 ],    value=2,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[6,7,8 ],    value=3,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[9 ],    value=4,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[10, 11,12],    value=5,    inplace=True)


print(final_df.dtypes.value_counts())

final_df.drop(final_df.index[final_df['CHOICE'] == 14], inplace = True)
final_df.drop(final_df.index[final_df['CHOICE'] == 13], inplace = True)


# ALTERNATIVE SPECIFIC CONSTANTS
ASC_PT = Beta('ASC_PT', 0, None, None, 0)
ASC_AUTO = Beta('ASC_AUTO', 0, None, None, 0)
ASC_Bike= Beta('ASC_Bike', 0, None, None, 0)
ASC_RideHailing = Beta('ASC_RideHailing', 0, None, None, 0)
ASC_Mixed_Modes = Beta('ASC_Mixed_Modes', 0, None, None, 0)

# BETA FOR INDIVIDUAL/USER CHARACTERSTICS
B_AGE = Beta('B_AGE', 0, None, None, 0)
B_Job_Status = Beta('B_Job_Status', 0, None, None, 0)
B_GENDER = Beta('B_GENDER', 0, None, None, 0)
B_WORKPLACE = Beta('B_WORKPLACE', 0, None, None, 0)
B_HOUSEHOLD = Beta('B_HOUSEHOLD', 0, None, None, 0)

# BETA FOR MODE SPECIFIC TRAVEL CHARACTERSTICS

# FOOT
B_TFreq_FOOT = Beta('B_TFreq_FOOT', 0, None, None, 0)
B_TDuration_FOOT = Beta('B_TDuration_FOOT', 0, None, None, 0)

#Bike
B_TFreq_Bike = Beta('B_TFreq_Bike', 0, None, None, 0)
B_TDurationBike = Beta('B_TDurationBike', 0, None, None, 0)

#PT
B_TFreq_PT = Beta('B_TFreq_PT', 0, None, None, 0)
B_TDurationPT = Beta('B_TDurationPT', 0, None, None, 0)

#AUTO
B_TFreq_AUTO = Beta('B_TFreq_AUTO', 0, None, None, 0)
B_TDurationAUTO = Beta('B_TDurationAUTO', 0, None, None, 0)

#RideHailing
B_TFreq_RideHailing = Beta('B_TFreq_RideHailing', 0, None, None, 0)
B_TDurationRideHailing = Beta('B_TDurationRideHailing', 0, None, None, 0)

#Mixed_Modes
B_TFreq_Mixed_Modes = Beta('B_TFreq_Mixed_Modes', 0, None, None, 0)
B_TDurationMixed_Modes = Beta('B_TDurationMixed_Modes', 0, None, None, 0)


V_FOOT  =   B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type  + B_TFreq_FOOT*Frequency
V_Bike  = ASC_Bike+  B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type +  B_TFreq_Bike*Frequency + B_TDurationBike *Duration
V_PT  =  B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type  + B_TDurationPT*Duration
V_AUTO  = ASC_AUTO+B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type  + B_TFreq_AUTO*Frequency + B_TDurationAUTO*Duration
V_RideHailing  = B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_TDurationRideHailing*Duration
V_Mixed_Modes  =   B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_TFreq_Mixed_Modes*Frequency


# Associate utility functions with the numbering of alternatives
V = {
     0: V_FOOT,
     1: V_Bike,
     2: V_PT,
     3:V_AUTO,
     4:V_RideHailing,
     5:V_Mixed_Modes,

     }



# The choice model is a mnl log logit
prob_FOOT = models.logit(V, None, 0)
prob_BIKE = models.logit(V, None,  1)
prob_PT = models.logit(V, None,  2)
prob_Auto = models.logit(V, None, 3)
prob_RideHailing = models.logit(V, None, 4)
prob_Mixed_Modes = models.logit(V, None,  5)



simulate = {'weight': Weight ,
            'prob_FOOT': prob_FOOT,
            'prob_BIKE': prob_BIKE,
            'prob_PT': prob_PT,
            'prob_Auto': prob_Auto,
            'prob_RideHailing': prob_RideHailing,
            'prob_Mixed_Modes': prob_Mixed_Modes,
           }



biogeme = bio.BIOGEME(database, simulate)

biogeme.modelName = 'BeforeMNLSimulation'
# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='C:/Users/artem/Desktop/ATS/theThesis/PythonOutput/Workplace/BeforeCovid/MNL/Before6_AlternativesMNL_weightV3.pickle')
except FileNotFoundError:
    sys.exit('Run first the script 01nestedEstimation.py in order to generate the file 01nestedEstimation.pickle.')
# simulatedValues is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulatedValues = biogeme.simulate(results.getBetaValues())
# Calculate confidence intervals
betas = biogeme.freeBetaNames
b = results.getBetasForSensitivityAnalysis(betas, size=100)
# Returns data frame containing, for each simulated value, the left
# and right bounds of the confidence interval calculated by
# simulation.
left, right = biogeme.confidenceIntervals(b, 0.95)
# We calculate now the market shares and their confidence intervals

simulatedValues['Weighted prob_FOOT'] =  simulatedValues['weight'] * simulatedValues['prob_FOOT']
left['Weighted prob_FOOT'] = left['weight'] * left['prob_FOOT']
right['Weighted prob_FOOT'] = right['weight'] * right['prob_FOOT']
marketShare_foot = simulatedValues['Weighted prob_FOOT'].mean()
marketShare_foot_left = left['Weighted prob_FOOT'].mean()
marketShare_foot_right = right['Weighted prob_FOOT'].mean()
print(f'Market share for foot: {100*marketShare_foot:.1f}% '
      f'[{100*marketShare_foot_left:.1f}%, {100*marketShare_foot_right:.1f}%]')




simulatedValues['Weighted prob_BIKE'] =  simulatedValues['weight'] * simulatedValues['prob_BIKE']
left['Weighted prob_BIKE'] = left['weight'] * left['prob_BIKE']
right['Weighted prob_BIKE'] = right['weight'] * right['prob_BIKE']
marketShare_bike = simulatedValues['Weighted prob_BIKE'].mean()
marketShare_bike_left = left['Weighted prob_BIKE'].mean()
marketShare_bike_right = right['Weighted prob_BIKE'].mean()
print(f'Market share for bike: {100*marketShare_bike:.1f}% '
      f'[{100*marketShare_bike_left:.1f}%, {100*marketShare_bike_right:.1f}%]')


simulatedValues['Weighted prob_PT'] =  simulatedValues['weight'] * simulatedValues['prob_PT']
left['Weighted prob_PT'] = left['weight'] * left['prob_PT']
right['Weighted prob_PT'] = right['weight'] * right['prob_PT']
marketShare_PT = simulatedValues['Weighted prob_PT'].mean()
marketShare_PT_left = left['Weighted prob_PT'].mean()
marketShare_PT_right = right['Weighted prob_PT'].mean()
print(f'Market share for PT: {100*marketShare_PT:.1f}% '
      f'[{100*marketShare_PT_left:.1f}%, {100*marketShare_PT_right:.1f}%]')


simulatedValues['Weighted prob_Auto'] =  simulatedValues['weight'] * simulatedValues['prob_Auto']
left['Weighted prob_Auto'] = left['weight'] * left['prob_Auto']
right['Weighted prob_Auto'] = right['weight'] * right['prob_Auto']
marketShare_Auto = simulatedValues['Weighted prob_Auto'].mean()
marketShare_Auto_left = left['Weighted prob_Auto'].mean()
marketShare_Auto_right = right['Weighted prob_Auto'].mean()
print(f'Market share for Auto: {100*marketShare_Auto:.1f}% '
      f'[{100*marketShare_Auto_left:.1f}%, {100*marketShare_Auto_right:.1f}%]')


simulatedValues['Weighted prob_RideHailing'] =  simulatedValues['weight'] * simulatedValues['prob_RideHailing']
left['Weighted prob_RideHailing'] = left['weight'] * left['prob_RideHailing']
right['Weighted prob_RideHailing'] = right['weight'] * right['prob_RideHailing']
marketShare_RideHailing = simulatedValues['Weighted prob_RideHailing'].mean()
marketShare_RideHailing_left = left['Weighted prob_RideHailing'].mean()
marketShare_RideHailing_right = right['Weighted prob_RideHailing'].mean()
print(f'Market share for RideHailing: {100*marketShare_RideHailing:.1f}% '
      f'[{100*marketShare_RideHailing_left:.1f}%, {100*marketShare_RideHailing_right:.1f}%]')


simulatedValues['Weighted prob_Mixed_Modes'] =  simulatedValues['weight'] * simulatedValues['prob_Mixed_Modes']
left['Weighted prob_Mixed_Modes'] = left['weight'] * left['prob_Mixed_Modes']
right['Weighted prob_Mixed_Modes'] = right['weight'] * right['prob_Mixed_Modes']
marketShare_Mixed_Modes = simulatedValues['Weighted prob_Mixed_Modes'].mean()
marketShare_Mixed_Modes_left = left['Weighted prob_Mixed_Modes'].mean()
marketShare_Mixed_Modes_right = right['Weighted prob_Mixed_Modes'].mean()
print(f'Market share for Mixed_Modes: {100*marketShare_Mixed_Modes:.1f}% '
      f'[{100*marketShare_Mixed_Modes_left:.1f}%, {100*marketShare_Mixed_Modes_right:.1f}%]')
