import pandas as pd
import numpy as np
import datetime

import os
import sys
import pathlib
import glob
from sklearn.preprocessing import MinMaxScaler
import tweedie

# 0.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
TRIP_FEATURE_DATA_FILE = DATA_FILE + '/trip_feature'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
RESULT_FILE = BASE_FILE + '/result'

clean_risk_feature_file = RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv'
clean_risk_feature_df = pd.read_csv(clean_risk_feature_file)
clean_risk_feature_df = clean_risk_feature_df.set_index('vin')

# 1.create score
clean_risk_feature_s_df = clean_risk_feature_df[clean_risk_feature_df['vehicle_type'] == 's']
clean_risk_feature_y_df = clean_risk_feature_df[clean_risk_feature_df['vehicle_type'] == 'y']
clean_risk_feature_num = clean_risk_feature_df.select_dtypes(exclude=['object'])
clean_risk_feature_s_num = clean_risk_feature_s_df.select_dtypes(exclude=['object'])
clean_risk_feature_y_num = clean_risk_feature_y_df.select_dtypes(exclude=['object'])
scaler = MinMaxScaler()
scale_risk_feature_df = pd.DataFrame(scaler.fit_transform(clean_risk_feature_num),
                                     columns=clean_risk_feature_num.columns,
                                     index=clean_risk_feature_num.index)
scale_risk_feature_s_df = pd.DataFrame(scaler.transform(clean_risk_feature_s_num),
                                       columns=clean_risk_feature_s_num.columns,
                                       index=clean_risk_feature_s_num.index)
scale_risk_feature_y_df = pd.DataFrame(scaler.transform(clean_risk_feature_y_num),
                                       columns=clean_risk_feature_y_num.columns,
                                       index=clean_risk_feature_y_num.index)

weight_df = pd.DataFrame({'warnCountPhk': 8,
                          'tripNum': 8,
                          'normL1':	8,
                          'qdcPerTrip': 8,
                          'chargeNum': 8,
                          'overChargeTime': 8,
                          'brakeMedian': 4,
                          'averageTripCurve': 4,
                          'distanceMedian': 4,
                          'durationMedian': 4,
                          'avgSpd': 4,
                          'lateNightDisRatio': 4,
                          '1stRouteTripRatio': 4,
                          'avgDurPerCharge': 4,
                          'ChargeDayNum': 4,
                          'duskDurationPerDay': 2,
                          'longTripRatio': 2,
                          'bdPhk': 0,
                          'distance': 0,
                          'duration': 0,
                          'disPerTrip': 0,
                          'activeDay': 0,
                          'activeDayRatio': 0,
                          'mileage': 0,
                          'distancePerDay': 0,
                          'durationPerDay': 0,
                          'tripNumPerDay': 0,
                          'lowSpdDurationPerDay': 0,
                          'afternoonDisRatio': 0,
                          'afternoonDurationPerDay': 0,
                          'afternoonDurationPerTrip': 0,
                          'duskDisRatio': 0,
                          'duskDurationPerTrip'	: 0,
                          'lateNightDurationPerDay'	: 0,
                          'lateNightDurationPerTrip': 0,
                          'lateNightTripRatio': 0,
                          'highCurveTripRatio': 0,
                          'longDisTripRatio': 0,
                          'holidayTripRatio': 0,
                          'weekendTripRatio': 0,
                          '2ndRouteTripRatio': 0,
                          '3rdRouteTripRatio': 0,
                          'tripDistEntropy': 0,
                          'tripMissCount': 0,
                          'missMileage': 0,
                          'qdcPhk': 0,
                          'startSoc': 0,
                          'endSoc': 0,
                          'lowQTripNum': 0,
                          'fullQTripNum': 0,
                          'chargeStartSoc': 0,
                          'chargeEndSoc': 0,
                          'overChargeDur': 0,
                          'qcPerCharge': 0,
                          },
                         index=[1])
insurance_df = np.exp(scale_risk_feature_df.dot(weight_df.transpose()))
score = insurance_df
score.columns = ['score']
score.to_csv(RESULT_FILE+'/score.csv')
insurance_s_df = np.exp(scale_risk_feature_s_df.dot(weight_df.transpose()))
insurance_y_df = np.exp(scale_risk_feature_y_df.dot(weight_df.transpose()))

# 2.create insurance data
from scipy.stats import poisson
insurance_s_df.columns = ['y_bar']
# insurance_s_df['position'] = (insurance_s_df['y_bar'].rank()-1)/len(insurance_s_df)
insurance_s_df['position'] = insurance_s_df['y_bar'].rank(pct=True)
mu_s = 0.27
# insurance_s_df['y'] = np.random.poisson(2.6, len(insurance_s_df))\
insurance_s_df['y'] = insurance_s_df['position'].apply(lambda x: poisson.ppf(x, mu_s))
insurance_s_df['y'] = insurance_s_df['y'].replace(np.inf, 10)

insurance_y_df.columns = ['y_bar']
# insurance_y_df['position'] = (insurance_y_df['y_bar'].rank()-1)/len(insurance_y_df)
insurance_y_df['position'] = insurance_y_df['y_bar'].rank(pct=True)
mu_y = 0.36
# insurance_s_df['y'] = np.random.poisson(2.6, len(insurance_s_df))\
insurance_y_df['y'] = insurance_y_df['position'].apply(lambda x: poisson.ppf(x, mu_y))
insurance_y_df['y'] = insurance_y_df['y'].replace(np.inf, 10)

# insurance_df['y'] = insurance_df['y_bar'].\
#     apply(lambda x: tweedie.tweedie(mu=x, p=1.6, phi=20).rvs(1).tolist()[0])

insurance_df = pd.concat([insurance_s_df, insurance_y_df], axis=0)
insurance_df.to_csv(DATA_FILE + '/insurance.csv')


