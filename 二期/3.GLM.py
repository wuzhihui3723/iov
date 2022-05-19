import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
import re

# 1.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
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
insurance_df = insurance_df['y']

# merge insurance data and binned risk feature
data = bin_risk_feature_df.merge(insurance_df,
                                 how='inner',
                                 on='vin')

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
# 数据集按照二八比例分为验证集和训练集
X = data.iloc[:, :-1]
drop_col_corr = ['warnCountPhk', 'activeDay', 'mileage', 'distancePerDay',
                 'lateNightTripRatio', 'highCurveTripRatio', 'secondRouteTripRatio',
                 'tripMissCount', 'missMileage', 'speedingPhk', 'provinceHighwayDisRatio']
X_drop_corr = X.drop(columns=drop_col_corr)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X_drop_corr,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
train_data = X_train.merge(y_train,
                           how='left',
                           on='vin')
# construct GLM1
formula = 'y ~ C(' + ',Treatment)+C('.join(X_train.columns) + ',Treatment)'
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
keep_col = ['bdPhk', 'distance', 'durationMedian', 'tripNum',
            'disPerTrip', 'activeDayRatio', 'tripNumPerDay',
            'normL1', 'duskDurationPerDay', 'lateNightDisRatio',
            'lateNightDurationPerDay', 'lateNightDurationPerTrip',
            'longDisTripRatio', 'firstRouteTripRatio', 'thirdRouteTripRatio',
            'tripDistEntropy', 'qdcPerTrip', 'qdcPhk', 'endSoc',
            'chargeNum', 'chargeStartSoc', 'avgDurPerCharge',
            'overChargeDur', 'ChargeDayNum', 'qcPerCharge',
            'speedingDurationPerDay', 'vehicle_type']
X_keep_col = X[keep_col]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_keep_col,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
train_data2 = X_train2.merge(y_train2,
                             how='left',
                             on='vin')
formula2 = 'y ~ C(' + ',Treatment)+C('.join(X_train2.columns) + ',Treatment)'
train_model2 = smf.glm(formula=formula2,
                       data=train_data2,
                       family=sm.families.Poisson(link=sm.families.links.log))
train_result2 = train_model2.fit()
print(train_result2.summary())

# save the model to disk
model_filename = RESULT_FILE + '/finalized_model.sav'
# model_filename_pkl = RESULT_FILE + '/finalized_model.pkl'
train_result2.save(model_filename)
# train_result2.save(model_filename_pkl, remove_data=True)
# X_test2.to_csv(RESULT_FILE + '/finalized_model_data.sav')
# model_filename = RESULT_FILE + '/finalized_model.sav'
# pickle.dump(train_model2, open(model_filename, 'wb'))

# 3.save coefficients and p_values
params_df2 = pd.DataFrame({'params': train_result2.params,
                           'p_values': train_result2.pvalues
                           })
params_df2.to_csv(RESULT_FILE + '/GLM2_params.csv')

# 4.prediction on test data
# test_prediction = pd.DataFrame(mod1.predict(X_test),
#                                columns=['prediction'])
test_prediction = pd.DataFrame(train_result2.predict(X_test2),
                               columns=['prediction'])

test_data = test_prediction.merge(y_test2,
                                  how='left',
                                  on='vin'). \
    merge(bin_risk_feature_df, how='left', on='vin'). \
    merge(clean_risk_feature_df, how='left', on='vin', suffixes=['_bin', '_clean'])

# 5.lift curve by y
lift_curve_df = test_data.sort_values('y')
# lift_curve_df['y_group'] = pd.qcut(lift_curve_df['prediction'], 10, duplicates='drop')
bins = pd.IntervalIndex.from_tuples([(0, 0.1), (0.1, 0.2), (0.2, 0.3),
                                     (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                                     (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
                                     (0.9, 10)])
lift_curve_df['y_group'] = pd.cut(lift_curve_df['prediction'], bins)
lift_curve_result_df = lift_curve_df.groupby(['y_group']). \
    agg({'y': 'mean',
         'prediction': 'mean',
         }).fillna(0)
# lift_curve_result_df.index = lift_curve_result_df.index.astype(str)
lift_curve_result_df.to_csv(RESULT_FILE + '/lift_curve/lift_curve_y.csv', index=False)
lift_curve_result_df_plot = lift_curve_result_df.reset_index()


def plot_lift_curve(df):
    ax.scatter(df.index.astype(str), df.y,
               linewidths=2, label='real_value')
    ax.plot(df.index.astype(str), df.y)
    ax.scatter(df.index.astype(str), df.prediction,
               linewidths=2, label='prediction')
    ax.plot(df.index.astype(str), df.prediction)
    ax.legend()


fig, ax = plt.subplots()
sns.set()
plot_lift_curve(lift_curve_result_df)
ax.legend()
plt.xticks(rotation=90)
plt.title("real_value vs prediction")
fig.tight_layout()
fig.savefig(RESULT_FILE + '/lift_plot/lift_curve_y.jpg', dpi=600)

# 6.lift curve by features
for i in keep_col:
    lift_curve_features = test_data.sort_values(i + '_clean')
    lift_curve_features_result = lift_curve_features.groupby([i + '_bin']). \
        agg({'y': 'mean',
             'prediction': 'mean',
             }).fillna(0)
    if i != "vehicle_type":
        lift_curve_features_result["sort"] = lift_curve_features_result.index
        lift_curve_features_result["sort"] = lift_curve_features_result["sort"]. \
            apply(lambda x: re.findall(r'[-+]?\d+[,.]?\d*', x)[0]). \
            astype("float64")
        lift_curve_features_result = lift_curve_features_result.sort_values("sort")
    lift_curve_features_result.to_csv(RESULT_FILE + '/lift_curve/lift_curve_' + i + '.csv')
    fig, ax = plt.subplots()
    sns.set()
    plot_lift_curve(lift_curve_features_result)
    ax.legend()
    plt.xticks(rotation=90)
    plt.title(i + ":real_value vs prediction")
    fig.tight_layout()
    fig.savefig(RESULT_FILE + '/lift_plot/' + 'lift_curve_' + i + '.jpg', dpi=600)

