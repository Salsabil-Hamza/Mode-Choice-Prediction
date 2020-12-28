import sys

import biogeme.database as db
import pandas as pd
import numpy as np
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, DefineVariable, bioMultSum
import biogeme.results as res




# Read the data
from matplotlib.sankey import Sankey

df = pd.read_csv('Results_All_Merged_Work.csv', sep = ';' )
pd.set_option('display.max_row', 13000)
pd.set_option('display.max_columns', 50)
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

database = db.Database('NL_model', final_df  )

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

final_df.drop(final_df.index[final_df['CHOICE'] == 14], inplace = True)
final_df.drop(final_df.index[final_df['CHOICE'] == 13], inplace = True)


# List of parameters to be estimated

# ALTERNATIVE SPECIFIC CONSTANTS

ASC_Bicycle = Beta('ASC_Bicycle', 0,-20,20, 0)
ASC_scooter= Beta('ASC_scooter', 0,-20,20, 0)
ASC_escooter= Beta('ASC_escooter', 0,-20,20, 0)
ASC_ebike= Beta('ASC_ebike', 0,-20,20, 0)
ASC_PT = Beta('ASC_PT', 0,-20,20, 0)
ASC_selfDriving = Beta('ASC_selfDriving', 0,-20,20, 0)
ASC_fampooling = Beta('ASC_fampooling', 0,-20,20, 0)
ASC_carpooling = Beta('ASC_carpooling', 0,-20,20, 0)
ASC_RideHailing = Beta('ASC_RideHailing', 0,-20,20, 0)
ASC_Parkandride = Beta('ASC_Parkandride', 0,-20,20, 0)
ASC_Kissandride = Beta('ASC_Kissandride', 0,-20,20, 0)
ASC_Bikeandride = Beta('ASC_Bikeandride', 0,-20,20, 0)


# BETA FOR INDIVIDUAL/USER CHARACTERSTICS
B_AGE = Beta('B_AGE', 0,-20,20, 0)
B_Job_Status = Beta('B_Job_Status', 0,-20,20, 0)
B_GENDER = Beta('B_GENDER', 0,-20,20, 0)
B_WORKPLACE = Beta('B_WORKPLACE', 0,-20,20, 0)
B_HOUSEHOLD = Beta('B_HOUSEHOLD', 0,-20,20, 0)

# BETA FOR MODE SPECIFIC TRAVEL CHARACTERSTICS

# FOOT
B_TFreq_FOOT = Beta('B_TFreq_FOOT', 0,-20,20, 0)
B_TDuration_FOOT = Beta('B_TDuration_FOOT', 0,-20,20, 0)

#Bicycle
B_TFreq_Bicycle = Beta('B_TFreq_Bicycle', 0,-20,20, 0)
B_TDuration_Bicycle = Beta('B_TDuration_Bicycle', 0,-20,20, 0)

#e-bike
B_TFreq_ebike = Beta('B_TFreq_ebike', 0,-20,20, 0)
B_TDuration_ebike = Beta('B_TDuration_ebike', 0,-20,20, 0)

#scooter
B_TFreq_scooter = Beta('B_TFreq_scooter', 0,-20,20, 0)
B_TDuration_scooter = Beta('B_TDuration_scooter', 0,-20,20, 0)

#escooter
B_TFreq_escooter = Beta('B_TFreq_escooter', 0,-20,20, 0)
B_TDuration_escooter = Beta('B_TDuration_escooter', 0,-20,20, 0)

#PT
B_TFreq_PT = Beta('B_TFreq_PT', 0,-20,20, 0)
B_TDurationPT = Beta('B_TDurationPT', 0,-20,20, 0)


#selfDriving
B_TFreq_selfDriving = Beta('B_TFreq_selfDriving', 0,-20,20, 0)
B_TDuration_selfDriving = Beta('B_TDuration_selfDriving', 0,-20,20, 0)

#fampooling
B_TFreq_fampooling = Beta('B_TFreq_fampooling', 0,-20,20, 0)
B_TDuration_fampooling = Beta('B_TDuration_fampooling', 0,-20,20, 0)

#carpooling
B_TFreq_carpooling = Beta('B_TFreq_carpooling', 0,-20,20, 0)
B_TDuration_carpooling = Beta('B_TDuration_carpooling', 0,-20,20, 0)

#RideHailing
B_TFreq_RideHailing = Beta('B_TFreq_RideHailing', 0,-20,20, 0)
B_TDuration_RideHailing = Beta('B_TDuration_RideHailing', 0,-20,20, 0)

#Parkandride
B_TFreq_Parkandride = Beta('B_TFreq_Parkandride', 0,-20,20, 0)
B_TDuration_Parkandride = Beta('B_TDuration_Parkandride', 0,-20,20, 0)

#Bikeandride
B_TFreq_Bikeandride = Beta('B_TFreq_Bikeandride', 0,-20,20, 0)
B_TDuration_Bikeandride = Beta('B_TDuration_Bikeandride', 0,-20,20, 0)

#Kissandride
B_TFreq_Kissandride = Beta('B_TFreq_Kissandride', 0,-20,20, 0)
B_TDuration_Kissandride = Beta('B_TDuration_Kissandride', 0,-20,20, 0)



V_FOOT  =    B_TFreq_FOOT*Frequency
V_Bicycle  = ASC_Bicycle+   B_TFreq_Bicycle*Frequency + B_TDuration_Bicycle *Duration
V_ebike  = ASC_ebike+   B_TFreq_ebike*Frequency + B_TDuration_ebike *Duration
V_scooter  =     B_TFreq_scooter*Frequency
V_escooter  =  0
V_PT =  B_TDurationPT*Duration

V_selfDriving  = ASC_selfDriving   + B_TDuration_selfDriving*Duration
V_fampooling  = ASC_fampooling  + B_TFreq_fampooling*Frequency
V_carpooling  =   B_Job_Status* Job_Status

V_RideHailing  =   0

V_Parkandride  =    0
V_Bikeandride  =   B_TFreq_Bikeandride*Frequency + B_TDuration_Bikeandride*Duration
V_Kissandride  =    B_TFreq_Kissandride*Frequency + B_TDuration_Kissandride*Duration
# Associate utility functions with the numbering of alternatives
V = {
     0: V_FOOT,

     1: V_Bicycle,
     2: V_ebike,
     3: V_scooter,
     4: V_escooter,

     5: V_PT,

     6:V_selfDriving,
     7:V_fampooling,
     8:V_carpooling,

     9:V_RideHailing,

     10:V_Parkandride,
     11:V_Bikeandride,
     12:V_Kissandride,
     }
#add maybe availability???

# Definition of the nests:
# 1: nests parameter
# 2: list of alternatives
MU_Bike = Beta('MU_Bike', 1.0, 1.0,10, 0)
MU_PT = Beta('MU_PT', 1.0, 1.0,10, 0)
MU_Auto= Beta('MU_Auto', 1.0, 1.0,10, 0)
MU_RideHailing= Beta('MU_RideHailing', 1.0, 1.0,10, 0)
MU_Mixed_Modes= Beta('MU_Mixed_Modes', 1.0, 1.0,10, 0)


Foot_NEST = 1.0, [0]
Bike_NEST= MU_Bike, [1,2,3,4]
PT_NEST = MU_PT, [5]
Auto_NEST = MU_Auto, [6,7,8]
RideHailing_NEST = MU_RideHailing,[9]
Mixed_Modes_NEST = MU_Mixed_Modes,[10, 11,12]
nests = Foot_NEST, Bike_NEST, PT_NEST, Auto_NEST, RideHailing_NEST, Mixed_Modes_NEST


# The choice model is a nested logit
prob_foot = models.lognested(V, None, nests, 0)
prob_bicycle = models.nested(V, None, nests, 1)
prob_ebike = models.nested(V, None, nests, 2)
prob_scooter = models.nested(V, None, nests, 3)
prob_escooter = models.nested(V, None, nests, 4)
prob_pt = models.nested(V, None, nests, 5)
prob_selfDriving = models.nested(V, None, nests, 6)
prob_fampooling = models.nested(V, None, nests, 7)
prob_carpooling = models.nested(V, None, nests, 8)
prob_RideHailing = models.nested(V, None, nests, 9)
prob_Parkandride = models.nested(V, None, nests, 10)
prob_Bikeandride = models.nested(V, None, nests, 11)
prob_Kissandride = models.nested(V, None, nests, 12)



simulate = {'weight': Weight,
            'Prob. foot': prob_foot,
            'Prob. bicycle': prob_bicycle,
            'Prob. ebike': prob_ebike,
'Prob. escooter': prob_escooter,
            'Prob. scooter': prob_scooter,
            'Prob. pt': prob_pt,
'Prob. selfDriving': prob_selfDriving,
            'Prob. fampooling':prob_fampooling,
            'Prob. carpooling': prob_carpooling,
'Prob. RideHailing': prob_RideHailing,
            'Prob. Parkandride': prob_Parkandride ,
            'Prob. Bikeandride': prob_Bikeandride ,
            'Prob. Kissandride': prob_Kissandride ,


           }

biogeme = bio.BIOGEME(database, simulate)
biogeme.modelName = '02nestedSimulation'
# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='C:/Users/artem/Desktop/ATS/theThesis/PythonOutput/Workplace/BeforeCovid/NL/logNestedFunction/Before_NL_6Alternatives_MuV5.pickle')
except FileNotFoundError:
    sys.exit('Run first the script 01nestedEstimation.py in order to generate the '
             'file 01nestedEstimation.pickle.')
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
#''''
simulatedValues['Weighted prob. foot'] = \
    simulatedValues['weight'] * simulatedValues['Prob. foot']
left['Weighted prob. foot'] = left['weight'] * left['Prob. foot']
right['Weighted prob. foot'] = right['weight'] * right['Prob. foot']
marketShare_foot = simulatedValues['Weighted prob. foot'].mean()
marketShare_foot_left = left['Weighted prob. foot'].mean()
marketShare_foot_right = right['Weighted prob. foot'].mean()
print(f'Market share for foot: {100*marketShare_foot:.1f}% '
      f'[{100*marketShare_foot_left:.1f}%, {100*marketShare_foot_right:.1f}%]')

#'''''
simulatedValues['Weighted Prob. bicycle'] = \
    simulatedValues['weight'] * simulatedValues['Prob. bicycle']
left['Weighted Prob. bicycle'] = left['weight'] * left['Prob. bicycle']
right['Weighted Prob. bicycle'] = right['weight'] * right['Prob. bicycle']
marketShare_bicycle = simulatedValues['Weighted Prob. bicycle'].mean()
marketShare_bicycle_left = left['Weighted Prob. bicycle'].mean()
marketShare_bicycle_right = right['Weighted Prob. bicycle'].mean()
print(f'Market share for bicycle: {100*marketShare_bicycle:.1f}% '
      f'[{100*marketShare_bicycle_left:.1f}%, {100*marketShare_bicycle_right:.1f}%]')


simulatedValues['Weighted Prob. ebike'] = \
    simulatedValues['weight'] * simulatedValues['Prob. ebike']
left['Weighted Prob. ebike'] = left['weight'] * left['Prob. ebike']
right['Weighted Prob. ebike'] = right['weight'] * right['Prob. ebike']
marketShare_ebike = simulatedValues['Weighted Prob. ebike'].mean()
marketShare_ebike_left = left['Weighted Prob. ebike'].mean()
marketShare_ebike_right = right['Weighted Prob. ebike'].mean()
print(f'Market share for ebike: {100*marketShare_ebike:.1f}% '
      f'[{100*marketShare_ebike_left:.1f}%, {100*marketShare_ebike_right:.1f}%]')


simulatedValues['Weighted Prob. scooter'] = \
    simulatedValues['weight'] * simulatedValues['Prob. scooter']
left['Weighted Prob. scooter'] = left['weight'] * left['Prob. scooter']
right['Weighted Prob. scooter'] = right['weight'] * right['Prob. scooter']
marketShare_scooter = simulatedValues['Weighted Prob. scooter'].mean()
marketShare_scooter_left = left['Weighted Prob. scooter'].mean()
marketShare_scooter_right = right['Weighted Prob. scooter'].mean()
print(f'Market share for scooter: {100*marketShare_scooter:.1f}% '
      f'[{100*marketShare_scooter_left:.1f}%, {100*marketShare_scooter_right:.1f}%]')

simulatedValues['Weighted Prob. escooter'] = \
    simulatedValues['weight'] * simulatedValues['Prob. escooter']
left['Weighted Prob. escooter'] = left['weight'] * left['Prob. escooter']
right['Weighted Prob. escooter'] = right['weight'] * right['Prob. escooter']
marketShare_escooter = simulatedValues['Weighted Prob. escooter'].mean()
marketShare_escooter_left = left['Weighted Prob. escooter'].mean()
marketShare_escooter_right = right['Weighted Prob. escooter'].mean()
print(f'Market share for escooter: {100*marketShare_escooter:.1f}% '
      f'[{100*marketShare_escooter_left:.1f}%, {100*marketShare_escooter_right:.1f}%]')



simulatedValues['Weighted Prob. pt'] = \
    simulatedValues['weight'] * simulatedValues['Prob. pt']
left['Weighted Prob. pt'] = left['weight'] * left['Prob. pt']
right['Weighted Prob. pt'] = right['weight'] * right['Prob. pt']
marketShare_pt = simulatedValues['Weighted Prob. pt'].mean()
marketShare_pt_left = left['Weighted Prob. pt'].mean()
marketShare_pt_right = right['Weighted Prob. pt'].mean()
print(f'Market share for pt: {100*marketShare_pt:.1f}% '
      f'[{100*marketShare_pt_left:.1f}%, {100*marketShare_pt_right:.1f}%]')


simulatedValues['Weighted Prob. selfDriving'] = \
    simulatedValues['weight'] * simulatedValues['Prob. selfDriving']
left['Weighted Prob. selfDriving'] = left['weight'] * left['Prob. selfDriving']
right['Weighted Prob. selfDriving'] = right['weight'] * right['Prob. selfDriving']
marketShare_selfDriving = simulatedValues['Weighted Prob. selfDriving'].mean()
marketShare_selfDriving_left = left['Weighted Prob. selfDriving'].mean()
marketShare_selfDriving_right = right['Weighted Prob. selfDriving'].mean()
print(f'Market share for selfDriving: {100*marketShare_selfDriving:.1f}% '
      f'[{100*marketShare_selfDriving_left:.1f}%, {100*marketShare_selfDriving_right:.1f}%]')




simulatedValues['Weighted Prob. fampooling'] = \
    simulatedValues['weight'] * simulatedValues['Prob. fampooling']
left['Weighted Prob. fampooling'] = left['weight'] * left['Prob. fampooling']
right['Weighted Prob. fampooling'] = right['weight'] * right['Prob. fampooling']
marketShare_fampooling = simulatedValues['Weighted Prob. fampooling'].mean()
marketShare_fampooling_left = left['Weighted Prob. fampooling'].mean()
marketShare_fampooling_right = right['Weighted Prob. fampooling'].mean()
print(f'Market share for fampooling: {100*marketShare_fampooling:.1f}% '
      f'[{100*marketShare_fampooling_left:.1f}%, {100*marketShare_fampooling_right:.1f}%]')




simulatedValues['Weighted Prob. carpooling'] = \
    simulatedValues['weight'] * simulatedValues['Prob. carpooling']
left['Weighted Prob. carpooling'] = left['weight'] * left['Prob. carpooling']
right['Weighted Prob. carpooling'] = right['weight'] * right['Prob. carpooling']
marketShare_carpooling = simulatedValues['Weighted Prob. carpooling'].mean()
marketShare_carpooling_left = left['Weighted Prob. carpooling'].mean()
marketShare_carpooling_right = right['Weighted Prob. carpooling'].mean()
print(f'Market share for carpooling: {100*marketShare_carpooling:.1f}% '
      f'[{100*marketShare_carpooling_left:.1f}%, {100*marketShare_carpooling_right:.1f}%]')


simulatedValues['Weighted Prob. RideHailing'] = \
    simulatedValues['weight'] * simulatedValues['Prob. RideHailing']
left['Weighted Prob. RideHailing'] = left['weight'] * left['Prob. RideHailing']
right['Weighted Prob. RideHailing'] = right['weight'] * right['Prob. RideHailing']
marketShare_RideHailing = simulatedValues['Weighted Prob. RideHailing'].mean()
marketShare_RideHailing_left = left['Weighted Prob. RideHailing'].mean()
marketShare_RideHailing_right = right['Weighted Prob. RideHailing'].mean()
print(f'Market share for RideHailing: {100*marketShare_RideHailing:.1f}% '
      f'[{100*marketShare_RideHailing_left:.1f}%, {100*marketShare_RideHailing_right:.1f}%]')




simulatedValues['Weighted Prob. Parkandride'] = \
    simulatedValues['weight'] * simulatedValues['Prob. Parkandride']
left['Weighted Prob. Parkandride'] = left['weight'] * left['Prob. Parkandride']
right['Weighted Prob. Parkandride'] = right['weight'] * right['Prob. Parkandride']
marketShare_Parkandride = simulatedValues['Weighted Prob. Parkandride'].mean()
marketShare_Parkandride_left = left['Weighted Prob. Parkandride'].mean()
marketShare_Parkandride_right = right['Weighted Prob. Parkandride'].mean()
print(f'Market share for Parkandride: {100*marketShare_Parkandride:.1f}% '
      f'[{100*marketShare_Parkandride_left:.1f}%, {100*marketShare_Parkandride_right:.1f}%]')



simulatedValues['Weighted Prob. Bikeandride'] = \
    simulatedValues['weight'] * simulatedValues['Prob. Bikeandride']
left['Weighted Prob. Bikeandride'] = left['weight'] * left['Prob. Bikeandride']
right['Weighted Prob. Bikeandride'] = right['weight'] * right['Prob. Bikeandride']
marketShare_Bikeandride = simulatedValues['Weighted Prob. Bikeandride'].mean()
marketShare_Bikeandride_left = left['Weighted Prob. Bikeandride'].mean()
marketShare_Bikeandride_right = right['Weighted Prob. Bikeandride'].mean()
print(f'Market share for Bikeandride: {100*marketShare_Bikeandride:.1f}% '
      f'[{100*marketShare_Bikeandride_left:.1f}%, {100*marketShare_Bikeandride_right:.1f}%]')




simulatedValues['Weighted Prob. Kissandride'] = \
    simulatedValues['weight'] * simulatedValues['Prob. Kissandride']
left['Weighted Prob. Kissandride'] = left['weight'] * left['Prob. Kissandride']
right['Weighted Prob. Kissandride'] = right['weight'] * right['Prob. Kissandride']
marketShare_Kissandride = simulatedValues['Weighted Prob. Kissandride'].mean()
marketShare_Kissandride_left = left['Weighted Prob. Kissandride'].mean()
marketShare_Kissandride_right = right['Weighted Prob. Kissandride'].mean()
print(f'Market share for Kissandride: {100*marketShare_Kissandride:.1f}% '
      f'[{100*marketShare_Kissandride_left:.1f}%, {100*marketShare_Kissandride_right:.1f}%]')









