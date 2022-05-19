import pandas as pd
import numpy as np
import os
import copy


# 0.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
TRIP_FEATURE_DATA_FILE = DATA_FILE + '/trip_feature'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
RESULT_FILE = BASE_FILE + '/result'

risk_feature_new_file = RISK_FEATURE_DATA_FILE + '/risk_feature_new.csv'
risk_feature_old_file = RISK_FEATURE_DATA_FILE + '/risk_feature_old.csv'
risk_feature_new_df = pd.read_csv(risk_feature_new_file)
risk_feature_old_df = pd.read_csv(risk_feature_old_file)
risk_feature_df = risk_feature_old_df.merge(risk_feature_new_df,
                                            on="vin",
                                            how='left')
risk_feature_df = risk_feature_df.set_index('vin')

# 1.calculate quantile
risk_feature_df = risk_feature_df[~risk_feature_df.isin(['nan', 'NaT']).any(axis=1)]
risk_feature_df = risk_feature_df.dropna()
quantiles_list = np.arange(0, 101)/100
quantile_df = risk_feature_df.quantile(quantiles_list)

# 2.filter feature
clean_risk_feature = copy.deepcopy(risk_feature_df)
clean_risk_feature = clean_risk_feature[clean_risk_feature['tripNum'] > 1]
clean_risk_feature.loc[clean_risk_feature['avgSpd'] > 120*1000/3600, 'avgSpd'] = 120*1000/3600
clean_risk_feature.loc[clean_risk_feature['disPerTrip'] > 120*24*1000, 'disPerTrip'] = 120*24*1000
clean_risk_feature.loc[clean_risk_feature['activeDayRatio'] > 1, 'activeDayRatio'] = 1
clean_risk_feature.loc[clean_risk_feature['mileage'] > 120*24*1000*365, 'mileage'] = 120*24*1000*365
clean_risk_feature.loc[clean_risk_feature['distancePerDay'] > 120*24*1000, 'distancePerDay'] = 120*24*1000
clean_risk_feature.loc[clean_risk_feature['durationPerDay'] > 24*3600, 'durationPerDay'] = 24*3600
clean_risk_feature.loc[clean_risk_feature['lowSpdDurationPerDay'] > 24*3600, 'lowSpdDurationPerDay'] = 24*3600
clean_risk_feature['afternoonDurationPerDay'] = clean_risk_feature['afternoonDurationPerDay'].\
    apply(lambda x: x if x <= 2*3600 else 2*3600)
clean_risk_feature['afternoonDurationPerTrip'] = clean_risk_feature['afternoonDurationPerTrip'].\
    apply(lambda x: x if x <= 2*2*3600 else 2*2*3600)
clean_risk_feature['duskDurationPerDay'] = clean_risk_feature['duskDurationPerDay'].\
    apply(lambda x: x if x <= 2*3600 else 2*3600)
clean_risk_feature['duskDurationPerTrip'] = clean_risk_feature['duskDurationPerTrip'].\
    apply(lambda x: x if x <= 2*2*3600 else 2*2*3600)
clean_risk_feature['lateNightDurationPerDay'] = clean_risk_feature['lateNightDurationPerDay'].\
    apply(lambda x: x if x <= 6*3600 else 6*3600)
clean_risk_feature['lateNightDurationPerTrip'] = clean_risk_feature['lateNightDurationPerTrip'].\
    apply(lambda x: x if x <= 2*6*3600 else 2*6*3600)
qdcPerTrip_median = clean_risk_feature['qdcPerTrip'].median()
clean_risk_feature['qdcPerTrip'] = clean_risk_feature['qdcPerTrip'].\
    apply(lambda x: x if x >= 0 else qdcPerTrip_median)
qdcPhk_median = clean_risk_feature['qdcPhk'].median()
clean_risk_feature['qdcPhk'] = clean_risk_feature['qdcPhk'].\
    apply(lambda x: x if x >= 0 else qdcPhk_median)
qcPerCharge_median = clean_risk_feature['qcPerCharge'].median()
clean_risk_feature['qcPerCharge'] = clean_risk_feature['qcPerCharge'].\
    apply(lambda x: x if x >= 0 else qcPerCharge_median)

# 3.calculate quantiles
clean_quantiles_list = np.arange(0, 101)/100
clean_mean_std_df = clean_risk_feature.describe()
clean_mean_std_df.to_csv(RESULT_FILE + '/mean_std.csv')
clean_quantile_df = clean_risk_feature.quantile(quantiles_list)
clean_quantile_df.to_csv(RESULT_FILE + '/quantile.csv')
clean_risk_feature.to_csv(RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv')

# 4.cut data by quantiles
clean_risk_feature_obj = clean_risk_feature.select_dtypes(include=['object'])
clean_risk_feature_num = clean_risk_feature.select_dtypes(exclude=['object'])
bin_num = 5
bin_risk_feature_df = clean_risk_feature_num.apply(lambda x: pd.qcut(x, bin_num, duplicates='drop'), axis=0)
bin_risk_feature_df = bin_risk_feature_df.merge(clean_risk_feature_obj, how='left', on='vin')
bin_risk_feature_df.to_csv(RISK_FEATURE_DATA_FILE + '/bin_risk_feature.csv')
