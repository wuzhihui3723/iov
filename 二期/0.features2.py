import pandas as pd
import numpy as np
import os
import glob
import ast

BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
TRIP_FEATURE_DATA_FILE = DATA_FILE + '/trip_feature'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
trip_data_files = glob.glob(TRIP_FEATURE_DATA_FILE + "/*/*")


# calculate risk features
def calculate_risk_features(trip_feature_df):
    """
        Helper function for calculate_risk_features
        Adjust input argument to the specified origin
        Parameters
        ----------
        trip_feature_df : pandas dataframe
            store trip features

        Returns
        -------
        risk_feature_df : pandas dataframe
            store risk features
    """
    # calculate activeDay
    trip_feature_df['activeDay'] = trip_feature_df['activeDay']. \
        apply(lambda x: ast.literal_eval(str(x)))

    # calculate normL1
    center_lat = np.mean([trip_feature_df['startLat'], trip_feature_df['endLat']])
    center_lon = np.mean([trip_feature_df['startLon'], trip_feature_df['endLon']])
    trip_feature_df['L1_distance'] = np.abs(trip_feature_df['startLat'] - center_lat) \
                                     + np.abs(trip_feature_df['startLon'] - center_lon) \
                                     + np.abs(trip_feature_df['endLat'] - center_lat) \
                                     + np.abs(trip_feature_df['endLon'] - center_lon)

    risk_feature_df = trip_feature_df.groupby(['vin']). \
        agg({'brakeCount': 'median', 'warnCount': 'sum', 'bdCount': 'sum',
             'TripCurve': 'mean', 'distance': ['sum', 'median'],
             'duration': ['sum', 'median'], 'tripId': 'count',
             })
    risk_feature_df.columns = ['brakeMedian', 'warnCountPhk', 'bdPhk',
                               'averageTripCurve',
                               'distance', 'distanceMedian',
                               'duration', 'durationMedian',
                               'tripNum',
                               ]
    risk_feature_df['activeDay'] = len(set(trip_feature_df['activeDay'].sum()))
    # 二期指标
    # over speeding
    risk_feature_df['badWeatherRatio'] = np.sum(trip_feature_df['badWeatherFlag']) / risk_feature_df['tripNum']
    risk_feature_df['speedingDisRatio'] = np.sum(trip_feature_df['spdFlag'] *
                                                 trip_feature_df['distance']) / \
                                          risk_feature_df['distance']
    risk_feature_df['speedingDurationPerDay'] = np.sum(trip_feature_df['spdFlag'] *
                                                       trip_feature_df['duration']) / \
                                                risk_feature_df['activeDay']
    risk_feature_df['speedingPhk'] = np.sum(trip_feature_df['spdCount']) / (risk_feature_df['distance'] / 1000) * 100
    # road type
    risk_feature_df['nationalHighwayDisRatio'] = np.sum(trip_feature_df['nationalHighwayDis']) * 1000 / \
                                                 risk_feature_df['distance']
    risk_feature_df['provinceHighwayDisRatio'] = np.sum(trip_feature_df['provinceHighwayDis']) * 1000 / \
                                                  risk_feature_df['distance']
    risk_feature_df['countyDisRatio'] = np.sum(trip_feature_df['countyDis']) * 1000 / \
                                        risk_feature_df['distance']
    risk_feature_df['countryDisRatio'] = np.sum(trip_feature_df['countryDis']) * 1000 / \
                                         risk_feature_df['distance']
    risk_feature_df['highwayDisRatio'] = np.sum(trip_feature_df['highwayDis']) * 1000 / \
                                         risk_feature_df['distance']
    risk_feature_df['cityHighWayDisRatio'] = np.sum(trip_feature_df['cityHighWayDis']) * 1000 / \
                                             risk_feature_df['distance']

    return risk_feature_df


risk_feature_dfs = []
for i in trip_data_files:
    vin = os.path.splitext(os.path.basename(i))[0]
    try:
        trip_feature_df = pd.read_csv(i)
    except UnicodeDecodeError:
        trip_feature_df = pd.read_csv(i, encoding="GB18030")
    trip_feature_df['vin'] = vin
    risk_feature_df = calculate_risk_features(trip_feature_df)
    risk_feature_dfs.append(risk_feature_df)

# 2.Concatenate all data into one DataFrame
RISK_FEATURE_DF = pd.concat(risk_feature_dfs)

# 3.save risk feature data
RISK_FEATURE_DF.to_csv(RISK_FEATURE_DATA_FILE + "/risk_feature_new.csv", index=False)
