
"""
 Estimation of a multinomial logit model, that will be used for simulation.
 There are mainly 5 alternatives: by foot, bicycle, Pt, car and mixed modes.
 The model includes all the available socioeconomic and modes related characteristics
 Based on RP data.
"""

import biogeme.database as db
import pandas as pd
import numpy as np
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta


df = pd.read_csv('C:/ATS/Results_All_Merged_Work.csv', sep = ';' )
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

normalizedWeight = np.ones(len(final_df['CHOICE']),dtype=float)
final_df['Weight'] = normalizedWeight

database = db.Database('MNL_model', final_df  )

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

#Removing some observations
final_df['CHOICE'].replace(    to_replace=[1,2,3,4 ],    value=1,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[5 ],    value=2,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[6,7,8 ],    value=3,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[9 ],    value=4,    inplace=True)
final_df['CHOICE'].replace(    to_replace=[10, 11,12],    value=5,    inplace=True)

final_df.drop(final_df.index[final_df['CHOICE'] == 14], inplace = True)
final_df.drop(final_df.index[final_df['CHOICE'] == 13], inplace = True)


# List of parameters to be estimated

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


V_FOOT  =  B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_FOOT*Frequency + B_TDuration_FOOT*Duration
V_Bike  = ASC_Bike+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_Bike*Frequency + B_TDurationBike *Duration
V_PT  = ASC_PT+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_PT*Frequency + B_TDurationPT*Duration
V_AUTO  = ASC_AUTO+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type +  B_Job_Status* Job_Status  + B_TFreq_AUTO*Frequency + B_TDurationAUTO*Duration
V_RideHailing  = ASC_RideHailing+  B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +B_TFreq_RideHailing*Frequency + B_TDurationRideHailing*Duration
V_Mixed_Modes  = ASC_Mixed_Modes+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_Mixed_Modes*Frequency + B_TDurationMixed_Modes*Duration


# Associate utility functions with the numbering of alternatives
V = {
     0: V_FOOT,
     1: V_Bike,
     2: V_PT,
     3:V_AUTO,
     4:V_RideHailing,
     5:V_Mixed_Modes,

     }


# Definition of the model. This is the contribution of each

logprob = models.loglogit(V, None, CHOICE)

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = 'Before6AlternativesMNL_weight_V1'

# Estimate the parameters
results = biogeme.estimate()

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)

