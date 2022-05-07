import pandas as pd
import numpy as np
import datetime
import os
import sys
import pathlib
import glob
import ast
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
# from sklearn.preprocessing import OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle

BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
clean_risk_feature_df = pd.read_csv(DATA_FILE+'/risk_feature/clean_risk_feature.csv')
# data = pd.read_csv(DATA_FILE+'/risk_feature/data.csv')
# data = pd.read_csv(DATA_FILE+'/risk_feature/demo_risk_feature.csv')
data = pd.read_csv(DATA_FILE+'/risk_feature/500car_risk_feature.csv')


# 1. 变量分组
data = data.rename(columns={"1stRouteTripRatio": "firstRouteTripRatio",
                            "2ndRouteTripRatio": "secondRouteTripRatio",
                            "3rdRouteTripRatio": "thirdRouteTripRatio",
                            })
clean_risk_feature_df = clean_risk_feature_df.rename(columns={"1stRouteTripRatio": "firstRouteTripRatio",
                                                              "2ndRouteTripRatio": "secondRouteTripRatio",
                                                              "3rdRouteTripRatio": "thirdRouteTripRatio",
                                                              })
keep_col = ['vin',
            'activeDayRatio', 'afternoonDurationPerTrip', 'averageTripCurve',
            'avgDurPerCharge', 'avgSpd', 'bdPhk',
            'ChargeDayNum', 'chargeNum', 'chargeStartSoc',
            'disPerTrip', 'distance', 'duration',
            'durationPerDay', 'duskDurationPerDay', 'endSoc',
            'fullQTripNum', 'lateNightDisRatio', 'qdcPerTrip',
            'qdcPhk', 'secondRouteTripRatio', 'thirdRouteTripRatio',
            'tripDistEntropy', 'tripNum', 'vehicle_type'
            ]
clean_risk_feature_df = clean_risk_feature_df[keep_col]
data = data[keep_col].fillna(0)

for i in clean_risk_feature_df.columns[1:-1]:
    # print(data[i])
    data.loc[data[i] > clean_risk_feature_df[i].max(), i] = clean_risk_feature_df[i].max()
    data.loc[data[i] < clean_risk_feature_df[i].min(), i] = clean_risk_feature_df[i].min()

bins_activeDayRatio = pd.IntervalIndex.from_tuples([(0.00479, 0.714), (0.714, 0.867),
                                                    (0.867, 0.948), (0.948, 0.988),
                                                    (0.988, 1.0)])
data['activeDayRatio'] = pd.cut(data['activeDayRatio'], bins_activeDayRatio)

bins_afternoonDurationPerTrip = pd.IntervalIndex.from_tuples([(9.999, 2174.115), (2174.115, 3375.185),
                                                              (3375.185, 4926.741), (4926.741, 9434.121),
                                                              (9434.121, 14400.0)])
data['afternoonDurationPerTrip'] = pd.cut(data['afternoonDurationPerTrip'], bins_afternoonDurationPerTrip)
bins_averageTripCurve = pd.IntervalIndex.from_tuples([(-0.000361, 16.082), (16.082, 41.345),
                                                      (41.345, 121.628), (121.628, 1113.815),
                                                      (1113.815, 376667.897)])
data['averageTripCurve'] = pd.cut(data['averageTripCurve'], bins_averageTripCurve)

bins_avgDurPerCharge = pd.IntervalIndex.from_tuples([(-0.001, 5526.667), (5526.667, 12777.348),
                                                     (12777.348, 20020.959), (20020.959, 34505.396),
                                                     (34505.396, 31366378.0)])
data['avgDurPerCharge'] = pd.cut(data['avgDurPerCharge'], bins_avgDurPerCharge)

bins_avgSpd = pd.IntervalIndex.from_tuples([(-0.00099034, 0.178), (0.178, 0.412),
                                            (0.412, 0.841), (0.841, 1.747),
                                            (1.747, 33.333)])
data['avgSpd'] = pd.cut(data['avgSpd'], bins_avgSpd)

bins_bdPhk = pd.IntervalIndex.from_tuples([(-0.001, 0.00534), (0.00534, 1.579),
                                           (1.579, 20.182), (20.182, 356730.258)])
data['bdPhk'] = pd.cut(data['bdPhk'], bins_bdPhk)

bins_ChargeDayNum = pd.IntervalIndex.from_tuples([(-0.001, 19.0), (19.0, 58.0),
                                                  (58.0, 98.0), (98.0, 161.0),
                                                  (161.0, 7289.0)])
data['ChargeDayNum'] = pd.cut(data['ChargeDayNum'], bins_ChargeDayNum)


bins_chargeNum = pd.IntervalIndex.from_tuples([(0.999, 18.0), (18.0, 69.0),
                                               (69.0, 139.0), (139.0, 259.0),
                                               (259.0, 57206.0)])
data['chargeNum'] = pd.cut(data['chargeNum'], bins_chargeNum)

bins_chargeStartSoc = pd.IntervalIndex.from_tuples([(-0.001, 30.885), (30.885, 42.75),
                                                    (42.75, 51.615), (51.615, 61.425),
                                                    (61.425, 100.0)])
data['chargeStartSoc'] = pd.cut(data['chargeStartSoc'], bins_chargeStartSoc)

bins_disPerTrip = pd.IntervalIndex.from_tuples([(2.331, 8107.714), (8107.714, 14471.999),
                                                (14471.999, 27603.624), (27603.624, 54688.918),
                                                (54688.918, 2880000.0)])
data['disPerTrip'] = pd.cut(data['disPerTrip'], bins_disPerTrip)

bins_distance = pd.IntervalIndex.from_tuples([(48.981, 1749152.777), (1749152.777, 4575194.798),
                                              (4575194.798, 8727112.534), (8727112.534, 16300758.71),
                                              (16300758.71, 19123684391.693)])
data['distance'] = pd.cut(data['distance'], bins_distance)

bins_duration = pd.IntervalIndex.from_tuples([(7651.999, 4182825.0), (4182825.0, 10916121.0),
                                              (10916121.0, 16908237.0), (16908237.0, 24612534.0),
                                              (24612534.0, 2223956253.0)])
data['duration'] = pd.cut(data['duration'], bins_duration)

bins_durationPerDay = pd.IntervalIndex.from_tuples([(1013.761, 37618.524), (37618.524, 61151.422),
                                                    (61151.422, 75232.804), (75232.804, 80724.754),
                                                    (80724.754, 86400.0)])
data['durationPerDay'] = pd.cut(data['durationPerDay'], bins_durationPerDay)

bins_duskDurationPerDay = pd.IntervalIndex.from_tuples([(0.235, 802.739), (802.739, 1509.903),
                                                        (1509.903, 2239.621), (2239.621, 3222.308),
                                                        (3222.308, 7200.0)])
data['duskDurationPerDay'] = pd.cut(data['duskDurationPerDay'], bins_duskDurationPerDay)

bins_endSoc = pd.IntervalIndex.from_tuples([(-0.001, 47.008), (47.008, 55.418),
                                            (55.418, 61.37), (61.37, 67.714),
                                            (67.714, 99.5)])
data['endSoc'] = pd.cut(data['endSoc'], bins_endSoc)

bins_fullQTripNum = pd.IntervalIndex.from_tuples([(-0.001, 4.0), (4.0, 24.0),
                                                  (24.0, 64.0), (64.0, 706.0)])
data['fullQTripNum'] = pd.cut(data['fullQTripNum'], bins_fullQTripNum)

data.loc[data["lateNightDisRatio"] > 0.999, 'lateNightDisRatio'] = 0.999
bins_lateNightDisRatio = pd.IntervalIndex.from_tuples([(-0.001, 0.0135), (0.0135, 0.0359),
                                                       (0.0359, 0.0777), (0.0777, 0.19),
                                                       (0.19, 0.999)])
data['lateNightDisRatio'] = pd.cut(data['lateNightDisRatio'], bins_lateNightDisRatio)

bins_qdcPerTrip = pd.IntervalIndex.from_tuples([(-0.001, 4.667), (4.667, 8.609),
                                                (8.609, 13.529), (13.529, 22.48),
                                                (22.48, 81.5)])
data['qdcPerTrip'] = pd.cut(data['qdcPerTrip'], bins_qdcPerTrip)


bins_qdcPhk = pd.IntervalIndex.from_tuples([(-0.001, 27.281), (27.281, 42.006),
                                            (42.006, 55.339), (55.339, 83.829),
                                            (83.829, 1614863.824)])
data['qdcPhk'] = pd.cut(data['qdcPhk'], bins_qdcPhk)


bins_secondRouteTripRatio = pd.IntervalIndex.from_tuples([(-0.000864, 0.00148), (0.00148, 0.00303),
                                                          (0.00303, 0.00658), (0.00658, 0.0164),
                                                          (0.0164, 0.5)])
data['secondRouteTripRatio'] = pd.cut(data['secondRouteTripRatio'], bins_secondRouteTripRatio)

bins_thirdRouteTripRatio = pd.IntervalIndex.from_tuples([(-0.001, 0.0014), (0.0014, 0.00279),
                                                         (0.00279, 0.0061), (0.0061, 0.0152),
                                                         (0.0152, 0.333)])
data['thirdRouteTripRatio'] = pd.cut(data['thirdRouteTripRatio'], bins_thirdRouteTripRatio)

bins_tripDistEntropy = pd.IntervalIndex.from_tuples([(0.19, 4.199), (4.199, 5.148),
                                                     (5.148, 5.974), (5.974, 6.671),
                                                     (6.671, 8.999)])
data['tripDistEntropy'] = pd.cut(data['tripDistEntropy'], bins_tripDistEntropy)

bins_tripNum = pd.IntervalIndex.from_tuples([(1.999, 69.0), (69.0, 176.0),
                                             (176.0, 400.0), (400.0, 802.0),
                                             (802.0, 8285.0)])
data['tripNum'] = pd.cut(data['tripNum'], bins_tripNum)

data[data.select_dtypes(['category']).columns] = data.select_dtypes(['category']).\
    apply(lambda x: x.astype(str))

# 2. load model
# load the model from disk
model_filename = RESULT_FILE + '/finalized_model.sav'
# finalized_model_data_filename = RESULT_FILE + '/finalized_model_data.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))
# finalized_model_data = pd.read_csv(finalized_model_data_filename)
# print(loaded_model.predict(data))

# 3. prediction model
prediction_df = pd.DataFrame({'vin': data.vin,
                              'prediction': loaded_model.predict(data)})
prediction_df = prediction_df.fillna(prediction_df['prediction'].mean())
prediction_df.to_csv(RESULT_FILE + '/save_model_prediction.csv', index=False)

