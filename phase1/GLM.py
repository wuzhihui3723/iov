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

# 1.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))#the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'

clean_risk_feature_file = RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv'
clean_risk_feature_df = pd.read_csv(clean_risk_feature_file)
clean_risk_feature_df = clean_risk_feature_df.set_index('vin')
clean_risk_feature_df = clean_risk_feature_df.rename(columns={"1stRouteTripRatio": "firstRouteTripRatio",
                                                              "2ndRouteTripRatio": "secondRouteTripRatio",
                                                              "3rdRouteTripRatio": "thirdRouteTripRatio",
                                                              })


bin_risk_feature_file = RISK_FEATURE_DATA_FILE + '/bin_risk_feature.csv'
bin_risk_feature_df = pd.read_csv(bin_risk_feature_file)
bin_risk_feature_df = bin_risk_feature_df.set_index('vin')
bin_risk_feature_df = bin_risk_feature_df.rename(columns={"1stRouteTripRatio": "firstRouteTripRatio",
                                                          "2ndRouteTripRatio": "secondRouteTripRatio",
                                                          "3rdRouteTripRatio": "thirdRouteTripRatio",
                                                          })

insurance_file = DATA_FILE + '/insurance.csv'
insurance_df = pd.read_csv(insurance_file).set_index('vin')

# data = bin_risk_feature_df.merge(insurance_df,
#                                  how='left',
#                                  on='vin').drop('y_bar', axis=1)

# 2. fit model
# GLM 1
# from patsy.contrasts import Treatment
# levels = [1,2,3,4]
# contrast = Treatment(reference=0).code_without_intercept(levels)
# print(contrast.matrix)

# formula = 'y ~ C(brakeMedian,Treatment)'
# mod1 = smf.glm(formula=formula,
#                data=data,
#                family=sm.families.Tweedie(var_power=1.6)).fit()
# print(mod1.summary())
X_train, X_test, y_train, y_test = train_test_split(bin_risk_feature_df,
                                                    insurance_df['y'],
                                                    test_size=0.2,
                                                    random_state=0)
train_data = X_train.merge(y_train,
                           how='left',
                           on='vin')
formula = 'y ~ C('+',Treatment)+C('.join(X_train.columns)+',Treatment)'
train_model = smf.glm(formula=formula,
                      data=train_data,
                      family=sm.families.Poisson(link=sm.families.links.log))
train_result = train_model.fit()
print(train_result.summary())
params_df = pd.DataFrame({'params': train_result.params,
                          'p_values': train_result.pvalues
                          })
params_df.to_csv(RESULT_FILE + '/GLM1_params.csv')

# GLM2
keep_col = ['activeDayRatio', 'afternoonDurationPerTrip', 'averageTripCurve',
            'avgDurPerCharge', 'avgSpd', 'bdPhk',
            'ChargeDayNum', 'chargeNum', 'chargeStartSoc',
            'disPerTrip', 'distance', 'duration',
            'durationPerDay', 'duskDurationPerDay', 'endSoc',
            'fullQTripNum', 'lateNightDisRatio', 'qdcPerTrip',
            'qdcPhk', 'secondRouteTripRatio', 'thirdRouteTripRatio',
            'tripDistEntropy', 'tripNum', 'vehicle_type'
            ]
bin_risk_feature_df2 = bin_risk_feature_df[keep_col]
X_train2, X_test2, y_train2, y_test2 = train_test_split(bin_risk_feature_df2,
                                                        insurance_df['y'],
                                                        test_size=0.2,
                                                        random_state=0)
train_data2 = X_train2.merge(y_train2,
                             how='left',
                             on='vin')
formula2 = 'y ~ C('+',Treatment)+C('.join(X_train2.columns)+',Treatment)'
train_model2 = smf.glm(formula=formula2,
                       data=train_data2,
                       family=sm.families.Poisson(link=sm.families.links.log))
train_result2 = train_model2.fit()
print(train_result2.summary())


# 3.save coef and p_values
params_df2 = pd.DataFrame({'params': train_result2.params,
                          'p_values': train_result2.pvalues
                          })
params_df2.to_csv(RESULT_FILE + '/GLM2_params.csv')

# 4.prediction on test data
# test_prediction = pd.DataFrame(mod1.predict(X_test),
#                                columns=['prediction'])
test_prediction = pd.DataFrame(train_result2.predict(X_test2),
                               columns=['prediction'])

test_data = test_prediction.merge(y_test,
                                  how='left',
                                  on='vin').\
    merge(bin_risk_feature_df, how='left', on='vin').\
    merge(clean_risk_feature_df, how='left', on='vin', suffixes=['_bin', '_clean'])

# 5.lift curve by y
lift_curve_df = test_data.sort_values('y')
# lift_curve_df['y_group'] = pd.qcut(lift_curve_df['prediction'], 10, duplicates='drop')
bins = pd.IntervalIndex.from_tuples([(0, 0.1), (0.1, 0.2), (0.2, 0.3),
                                     (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                     (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
                                     (0.9, 10)])
lift_curve_df['y_group'] = pd.cut(lift_curve_df['prediction'], bins)

lift_curve_result_df = lift_curve_df.groupby(['y_group']).\
    agg({'y': 'mean',
         'prediction': 'mean',
         }).fillna(0)
# lift_curve_result_df.index = lift_curve_result_df.index.astype(str)
lift_curve_result_df.to_csv(RESULT_FILE + '/lift_curve/lift_curve_y.csv')

# 6.lift curve by features
for i in keep_col:
    lift_curve_features = test_data.sort_values(i + '_clean')
    lift_curve_features_result = lift_curve_features.groupby([i + '_bin']). \
        agg({'y': 'mean',
             'prediction': 'mean',
             }).fillna(0)
    lift_curve_features_result.to_csv(RESULT_FILE + '/lift_curve/lift_curve_' + i + '.csv')



