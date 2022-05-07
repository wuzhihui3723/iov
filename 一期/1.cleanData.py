import pandas as pd
import numpy as np
import datetime

import os
import sys
import pathlib
import glob

# 执行执行该文件，
#
# 首先，该文件读取风险因子数据data/risk_feature/risk_feature.csv
#
# 其次，计算每个风险因子的1%分位数，并存储为result/quantile.csv
#
# 然后，清洗数据，生成data/risk_feature/clean_risk_feature.csv
#
# 最后，对每个风险因子等深分组，可通过bin_num修改分组数量，将生成的分组数据，生成data/risk_feature/bin_risk_feature.csv

# 0.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
TRIP_FEATURE_DATA_FILE = DATA_FILE + '/trip_feature'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
RESULT_FILE = BASE_FILE + '/result'

risk_feature_file = RISK_FEATURE_DATA_FILE + '/risk_feature.csv'
risk_feature_df = pd.read_csv(risk_feature_file)
risk_feature_df = risk_feature_df.set_index('vin')

# 1.calculate quantile
risk_feature_df = risk_feature_df[~risk_feature_df.isin(['nan', 'NaT']).any(axis=1)]
risk_feature_df = risk_feature_df.dropna()
quantiles_list = np.arange(0, 101)/100
quantile_df = risk_feature_df.quantile(quantiles_list)
#quantile_df.to_csv(RESULT_FILE + '/quantile.csv')

# 2.filter feature
# non_Ratio_cols = [col for col in risk_feature_df.columns if 'Ratio' not in col]
# non_Ratio_feature_df = risk_feature_df[non_Ratio_cols]
import copy
clean_risk_feature = copy.deepcopy(risk_feature_df)
clean_risk_feature = clean_risk_feature[clean_risk_feature['tripNum'] > 1]

clean_risk_feature.loc[clean_risk_feature['avgSpd'] > 120*1000/3600, 'avgSpd'] = 120*1000/3600
clean_risk_feature.loc[clean_risk_feature['disPerTrip'] > 120*24*1000, 'disPerTrip'] = 120*24*1000
clean_risk_feature.loc[clean_risk_feature['activeDayRatio'] > 1, 'activeDayRatio'] = 1
clean_risk_feature.loc[clean_risk_feature['mileage'] > 120*24*1000*365, 'mileage'] = 120*24*1000*365
clean_risk_feature.loc[clean_risk_feature['distancePerDay'] > 120*24*1000, 'distancePerDay'] = 120*24*1000
clean_risk_feature.loc[clean_risk_feature['durationPerDay'] > 24*3600, 'durationPerDay'] = 24*3600
clean_risk_feature.loc[clean_risk_feature['lowSpdDurationPerDay'] > 24*3600, 'lowSpdDurationPerDay'] = 24*3600
# clean_risk_feature['avgSpd'] = clean_risk_feature['avgSpd'].\
#     apply(lambda x: x if x <= 120*1000/3600 else 120*1000/3600)
# clean_risk_feature['disPerTrip'] = clean_risk_feature['disPerTrip'].\
#     apply(lambda x: x if x <= 120*24*1000 else 120*24*1000)
# clean_risk_feature['activeDayRatio'] = clean_risk_feature['activeDayRatio'].\
#     apply(lambda x: x if x <= 1 else 1)
# clean_risk_feature['mileage'] = clean_risk_feature['mileage'].\
#     apply(lambda x: x if x <= 120*24*1000*365 else 120*24*1000*36)
# clean_risk_feature['distancePerDay'] = clean_risk_feature['distancePerDay'].\
#     apply(lambda x: x if x <= 120*24*1000 else 120*24*1000)
# clean_risk_feature['durationPerDay'] = clean_risk_feature['durationPerDay'].\
#     apply(lambda x: x if x <= 24*3600 else 24*3600)
# clean_risk_feature['lowSpdDurationPerDay'] = clean_risk_feature['lowSpdDurationPerDay'].\
#     apply(lambda x: x if x <= 24*3600 else 24*3600)

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

clean_quantiles_list = np.arange(0, 101)/100
clean_mean_std_df = clean_risk_feature.describe()
clean_mean_std_df.to_csv(RESULT_FILE + '/mean_std.csv')
clean_quantile_df = clean_risk_feature.quantile(quantiles_list)
clean_quantile_df.to_csv(RESULT_FILE + '/quantile.csv')
clean_risk_feature.to_csv(RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv')

# 3.cut data by quantiles
clean_risk_feature_obj = clean_risk_feature.select_dtypes(include=['object'])
clean_risk_feature_num = clean_risk_feature.select_dtypes(exclude=['object'])
bin_num = 5
bin_risk_feature_df = clean_risk_feature_num.apply(lambda x: pd.qcut(x, bin_num, duplicates='drop'), axis=0)
bin_risk_feature_df = bin_risk_feature_df.merge(clean_risk_feature_obj, how='left', on='vin')
bin_risk_feature_df.to_csv(RISK_FEATURE_DATA_FILE + '/bin_risk_feature.csv')
