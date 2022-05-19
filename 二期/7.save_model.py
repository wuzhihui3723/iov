import pandas as pd
import os
import pickle

BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
clean_risk_feature_df = pd.read_csv(DATA_FILE + '/risk_feature/clean_risk_feature.csv')
data = pd.read_csv(DATA_FILE + '/risk_feature/data.csv')
# data = pd.read_csv(DATA_FILE+'/risk_feature/demo_risk_feature_correct.csv')
# data = pd.read_csv(DATA_FILE+'/risk_feature/500car_risk_feature.csv')

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
            'bdPhk', 'distance', 'durationMedian', 'tripNum',
            'disPerTrip', 'activeDayRatio', 'tripNumPerDay',
            'normL1', 'duskDurationPerDay', 'lateNightDisRatio',
            'lateNightDurationPerDay', 'lateNightDurationPerTrip',
            'longDisTripRatio', 'firstRouteTripRatio', 'thirdRouteTripRatio',
            'tripDistEntropy', 'qdcPerTrip', 'qdcPhk', 'endSoc',
            'chargeNum', 'chargeStartSoc', 'avgDurPerCharge',
            'overChargeDur', 'ChargeDayNum', 'qcPerCharge',
            'speedingDurationPerDay', 'vehicle_type']
clean_risk_feature_df = clean_risk_feature_df[keep_col]
data = data[keep_col]
# data = data[keep_col].fillna(0)

for i in clean_risk_feature_df.columns[1:-1]:
    # print(data[i])
    data.loc[data[i] > clean_risk_feature_df[i].max(), i] = clean_risk_feature_df[i].max() - 0.00000000001
    data.loc[data[i] < clean_risk_feature_df[i].min(), i] = clean_risk_feature_df[i].min() + 0.00000000001

bins_bdPhk = pd.IntervalIndex.from_tuples([(-0.001, 0.00689), (0.00689, 1.972),
                                           (1.972, 26.309), (26.309, 356730.258)])
data['bdPhk'] = pd.cut(data['bdPhk'], bins_bdPhk)
# 'distance'
bins_distance = pd.IntervalIndex.from_tuples([(48.981, 1802671.275), (1802671.275, 4612157.866),
                                              (4612157.866, 8824791.244), (8824791.244, 16447482.686),
                                              (16447482.686, 9320628079.421)])
data['distance'] = pd.cut(data['distance'], bins_distance)

# 'durationMedian',
bins_durationMedian = pd.IntervalIndex.from_tuples([(9.999, 1531.2), (1531.2, 3287.2),
                                                    (3287.2, 8746.4), (8746.4, 69166.0),
                                                    (69166.0, 12151894.0), ])
data['durationMedian'] = pd.cut(data['durationMedian'], bins_durationMedian)
# 'tripNum',
bins_tripNum = pd.IntervalIndex.from_tuples([(1.999, 65.0), (65.0, 164.0),
                                             (164.0, 388.0), (388.0, 807.0),
                                             (807.0, 8285.0), ])
data['tripNum'] = pd.cut(data['tripNum'], bins_tripNum)
# 'disPerTrip', \
bins_disPerTrip = pd.IntervalIndex.from_tuples([(2.331, 8130.964), (8130.964, 15253.078),
                                                (15253.078, 28535.334), (28535.334, 56727.685),
                                                (56727.685, 2880000.0), ])
data['disPerTrip'] = pd.cut(data['disPerTrip'], bins_disPerTrip)
# 'activeDayRatio'
bins_activeDayRatio = pd.IntervalIndex.from_tuples([(0.0168, 0.744), (0.744, 0.879),
                                                    (0.879, 0.951), (0.951, 0.989),
                                                    (0.989, 1.0), ])
data['activeDayRatio'] = pd.cut(data['activeDayRatio'], bins_activeDayRatio)
# 'tripNumPerDay',
bins_tripNumPerDay = pd.IntervalIndex.from_tuples([(0.00609, 0.369), (0.369, 0.991),
                                                   (0.991, 2.177), (2.177, 3.655),
                                                   (3.655, 81.0)])
data['tripNumPerDay'] = pd.cut(data['tripNumPerDay'], bins_tripNumPerDay)
#             'normL1',\
bins_normL1 = pd.IntervalIndex.from_tuples([(1.3940000000000001, 28185.661), (28185.661, 60402.757),
                                            (60402.757, 124149.421),
                                            (124149.421, 217249.103), (217249.103, 18378924.281),
                                            ])
data['normL1'] = pd.cut(data['normL1'], bins_normL1)
#             'duskDurationPerDay', \
bins_duskDurationPerDay = pd.IntervalIndex.from_tuples([(0.235, 818.527), (818.527, 1575.997),
                                                        (1575.997, 2299.887), (2299.887, 3263.907),
                                                        (3263.907, 7200.0),
                                                        ])
data['duskDurationPerDay'] = pd.cut(data['duskDurationPerDay'], bins_duskDurationPerDay)
#             'lateNightDisRatio',
bins_lateNightDisRatio = pd.IntervalIndex.from_tuples([(-0.001, 0.0138), (0.0138, 0.0359),
                                                       (0.0359, 0.0759), (0.0759, 0.188),
                                                       (0.188, 0.999)])
data['lateNightDisRatio'] = pd.cut(data['lateNightDisRatio'], bins_lateNightDisRatio)
#             'lateNightDurationPerDay',\
bins_lateNightDurationPerDay = pd.IntervalIndex.from_tuples([(0.414, 5608.401), (5608.401, 10921.197),
                                                             (10921.197, 16254.88), (16254.88, 21600.0), ])
data['lateNightDurationPerDay'] = pd.cut(data['lateNightDurationPerDay'], bins_lateNightDurationPerDay)
#             'lateNightDurationPerTrip'
bins_lateNightDurationPerTrip = pd.IntervalIndex.from_tuples([(79.999, 16110.222), (16110.222, 34932.388),
                                                              (34932.388, 43200.0)])
data['lateNightDurationPerTrip'] = pd.cut(data['lateNightDurationPerTrip'], bins_lateNightDurationPerTrip)
#  'longDisTripRatio'
bins_longDisTripRatio = pd.IntervalIndex.from_tuples([(-0.001, 0.00235), (0.00235, 0.0263),
                                                      (0.0263, 1.0)])
data['longDisTripRatio'] = pd.cut(data['longDisTripRatio'], bins_longDisTripRatio)
# 'firstRouteTripRatio'
bins_firstRouteTripRatio = pd.IntervalIndex.from_tuples([(-0.000846, 0.00153), (0.00153, 0.00321),
                                                         (0.00321, 0.00773), (0.00773, 0.02),
                                                         (0.02, 0.974), ])
data['firstRouteTripRatio'] = pd.cut(data['firstRouteTripRatio'], bins_firstRouteTripRatio)
#             'thirdRouteTripRatio',
bins_thirdRouteTripRatio = pd.IntervalIndex.from_tuples([(-0.001, 0.00135), (0.00135, 0.00274),
                                                         (0.00274, 0.00641), (0.00641, 0.0159),
                                                         (0.0159, 0.333), ])
data['thirdRouteTripRatio'] = pd.cut(data['thirdRouteTripRatio'], bins_thirdRouteTripRatio)
#             'tripDistEntropy', \
bins_tripDistEntropy = pd.IntervalIndex.from_tuples([(0.19, 4.145), (4.145, 5.081),
                                                     (5.081, 5.946), (5.946, 6.686),
                                                     (6.686, 8.999)])
data['tripDistEntropy'] = pd.cut(data['tripDistEntropy'], bins_tripDistEntropy)
#             'qdcPerTrip',\
bins_qdcPerTrip = pd.IntervalIndex.from_tuples([(-0.001, 4.857), (4.857, 9.192),
                                                (9.192, 14.523), (14.523, 23.241),
                                                (23.241, 81.5), ])
data['qdcPerTrip'] = pd.cut(data['qdcPerTrip'], bins_qdcPerTrip)
#             'qdcPhk',\
bins_qdcPhk = pd.IntervalIndex.from_tuples([(-0.001, 26.751), (26.751, 42.392),
                                            (42.392, 55.557), (55.557, 82.404),
                                            (82.404, 1614863.824), ])
data['qdcPhk'] = pd.cut(data['qdcPhk'], bins_qdcPhk)
#             'endSoc',
bins_endSoc = pd.IntervalIndex.from_tuples([(-0.001, 46.861), (46.861, 55.541),
                                            (55.541, 61.57), (61.57, 67.815),
                                            (67.815, 99.5), ])
data['endSoc'] = pd.cut(data['endSoc'], bins_endSoc)
#             'chargeNum', \
bins_chargeNum = pd.IntervalIndex.from_tuples([(0.999, 19.0), (19.0, 68.0),
                                               (68.0, 135.0), (135.0, 237.0),
                                               (237.0, 57206.0), ])
data['chargeNum'] = pd.cut(data['chargeNum'], bins_chargeNum)
#             'chargeStartSoc', \
bins_chargeStartSoc = pd.IntervalIndex.from_tuples([(-0.001, 30.588), (30.588, 42.832),
                                                    (42.832, 51.753), (51.753, 61.602),
                                                    (61.602, 100.0), ])
data['chargeStartSoc'] = pd.cut(data['chargeStartSoc'], bins_chargeStartSoc)
# 'avgDurPerCharge',
bins_avgDurPerCharge = pd.IntervalIndex.from_tuples([(-0.001, 6622.64), (6622.64, 13579.963),
                                                     (13579.963, 20771.203), (20771.203, 35470.265),
                                                     (35470.265, 31366378.0), ])
data['avgDurPerCharge'] = pd.cut(data['avgDurPerCharge'], bins_avgDurPerCharge)
# 'overChargeDur'
bins_overChargeDur = pd.IntervalIndex.from_tuples([(-0.001, 80834.4), (80834.4, 394471.2),
                                                   (394471.2, 1109273.6), (1109273.6, 31113423.0), ])
data['overChargeDur'] = pd.cut(data['overChargeDur'], bins_overChargeDur)
# 'ChargeDayNum'
bins_ChargeDayNum = pd.IntervalIndex.from_tuples([(-0.001, 20.0), (20.0, 60.0),
                                                  (60.0, 101.0), (101.0, 162.0),
                                                  (162.0, 7289.0), ])
data['ChargeDayNum'] = pd.cut(data['ChargeDayNum'], bins_ChargeDayNum)
# 'qcPerCharge',
bins_qcPerCharge = pd.IntervalIndex.from_tuples([(-0.001, 894.369), (894.369, 1140.441),
                                                 (1140.441, 1313.574), (1313.574, 1602.758),
                                                 (1602.758, 2500.0), ])
data['qcPerCharge'] = pd.cut(data['qcPerCharge'], bins_qcPerCharge)
# 'speedingDurationPerDay'
bins_speedingDurationPerDay = pd.IntervalIndex.from_tuples([(-0.001, 42514.381), (42514.381, 67043.424),
                                                            (67043.424, 75923.477), (75923.477, 80688.891),
                                                            (80688.891, 182850.031), ])
data['speedingDurationPerDay'] = pd.cut(data['speedingDurationPerDay'], bins_speedingDurationPerDay)
# 'vehicle_type'

data[data.select_dtypes(['category']).columns] = data.select_dtypes(['category']). \
    apply(lambda x: x.astype(str))

# 2. load model
# load the model from disk
model_filename = RESULT_FILE + '/finalized_model.sav'
# model_filename_pkl = RESULT_FILE + '/finalized_model.pkl'
# finalized_model_data_filename = RESULT_FILE + '/finalized_model_data.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))
# loaded_model_pkl = pickle.load(open(model_filename, 'rb'))
# finalized_model_data = pd.read_csv(finalized_model_data_filename)
# print(loaded_model.predict(data))

# 3. prediction model
prediction_df = pd.DataFrame({'vin': data.vin,
                              'prediction': loaded_model.predict(data)})
prediction_df = prediction_df.fillna(prediction_df['prediction'].mean())
prediction_df.to_csv(RESULT_FILE + '/save_model_prediction.csv', index=False)
