
import biogeme.database as db
import pandas as pd
import numpy as np
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, DefineVariable, bioMultSum



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




V_FOOT  =  B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_FOOT*Frequency + B_TDuration_FOOT*Duration

V_Bicycle  = ASC_Bicycle+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_Bicycle*Frequency + B_TDuration_Bicycle *Duration
V_ebike  = ASC_ebike+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_ebike*Frequency + B_TDuration_ebike *Duration
V_scooter  = ASC_scooter+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_scooter*Frequency + B_TDuration_scooter *Duration
V_escooter  = ASC_escooter+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_escooter*Frequency + B_TDuration_escooter *Duration

V_PT = ASC_PT+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +  B_TFreq_PT*Frequency + B_TDurationPT*Duration

V_selfDriving  = ASC_selfDriving+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type +  B_Job_Status* Job_Status  + B_TFreq_selfDriving*Frequency + B_TDuration_selfDriving*Duration
V_fampooling  = ASC_fampooling+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type +  B_Job_Status* Job_Status  + B_TFreq_fampooling*Frequency + B_TDuration_fampooling*Duration
V_carpooling  = ASC_carpooling+B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type +  B_Job_Status* Job_Status  + B_TFreq_carpooling*Frequency + B_TDuration_carpooling*Duration

V_RideHailing  = ASC_RideHailing+  B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status +B_TFreq_RideHailing*Frequency + B_TDuration_RideHailing*Duration

V_Parkandride  = ASC_Parkandride+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_Parkandride*Frequency + B_TDuration_Parkandride*Duration
V_Bikeandride  = ASC_Bikeandride+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_Bikeandride*Frequency + B_TDuration_Bikeandride*Duration
V_Kissandride  = ASC_Kissandride+ B_AGE*Age+ B_GENDER *Gender+ B_WORKPLACE*Workplace+ B_HOUSEHOLD*Household_Type + B_Job_Status* Job_Status + B_TFreq_Kissandride*Frequency + B_TDuration_Kissandride*Duration

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


# Definition of the nests:
# 1: nests parameter
# 2: list of alternatives
MU_Bike = Beta('MU_Bike', 1.5, 1, 10, 0)
MU_PT = Beta('MU_PT', 1.5, 1, 10, 0)
MU_Auto= Beta('MU_Auto', 1.5, 1, 10, 0)
MU_RideHailing= Beta('MU_RideHailing',1.5, 1, 10, 0)
MU_Mixed_Modes= Beta('MU_Mixed_Modes', 1.5, 1, 10, 0)

Foot_NEST = 1.0, [0]
Bike_NEST= MU_Bike, [1,2,3,4]
PT_NEST = MU_PT, [5]
Auto_NEST = MU_Auto, [6,7,8]
RideHailing_NEST = MU_RideHailing,[9]
Mixed_Modes_NEST = MU_Mixed_Modes,[10, 11,12]
nests = Foot_NEST, Bike_NEST, PT_NEST, Auto_NEST, RideHailing_NEST, Mixed_Modes_NEST

# The choice model is a nested logit, with availability conditions
logprob = models.lognested(V, None, nests, CHOICE)

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = 'Before_NL_6Alternatives_modelV1'

# Estimate the parameters
results = biogeme.estimate(bootstrap=50)

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
