import pandas as pd
import numpy as np
import datetime

import os
import sys
import pathlib
import glob
import ast


#1.load data
#BASE_FILE = os.path.dirname(os.path.abspath(__file__))#the directory of the script being run
DATA_FILE = '/sdb/trip/stat/'
outputfolder = '/sdb/risk/'
folderlist = ['2020_1001-2000.csv','2020_2001-3000.csv']
#RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'

#trip_data_files = glob.glob(TRIP_FEATURE_DATA_FILE + "/*.csv")

# calculate risk features
def calculate_risk_features(trip_feature_df):
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

    risk_feature_df['avgSpd'] = risk_feature_df['distance'] / risk_feature_df['duration']
    risk_feature_df['disPerTrip'] = risk_feature_df['distance'] / risk_feature_df['tripNum']
    risk_feature_df['activeDay'] = len(set(trip_feature_df['activeDay'].sum()))
    risk_feature_df['tripNumPerDay'] = risk_feature_df['tripNum'] / risk_feature_df['activeDay']
    risk_feature_df['normL1'] = np.sum(trip_feature_df['L1_distance']) / (2 * trip_feature_df.shape[0])

    risk_feature_df['afternoonDisRatio'] = np.sum(trip_feature_df['afternoonDis']) / \
                                           risk_feature_df['distance']
    risk_feature_df['afternoonDurationPerDay'] = np.sum(trip_feature_df['afternoonDuration']) / \
                                                 risk_feature_df['activeDay']
    risk_feature_df['afternoonDurationPerTrip'] = np.sum(trip_feature_df['afternoonDuration']) / \
                                                  np.sum(trip_feature_df['afternoonFlag'])

    risk_feature_df['duskDisRatio'] = np.sum(trip_feature_df['duskDis']) / \
                                      risk_feature_df['distance']
    risk_feature_df['duskDurationPerDay'] = np.sum(trip_feature_df['duskDuration']) / \
                                            risk_feature_df['activeDay']
    risk_feature_df['duskDurationPerTrip'] = np.sum(trip_feature_df['duskDuration']) / \
                                             np.sum(trip_feature_df['duskFlag'])

    risk_feature_df['lateNightDisRatio'] = np.sum(trip_feature_df['lateNightDis']) / \
                                           risk_feature_df['distance']
    risk_feature_df['lateNightDurationPerDay'] = np.sum(trip_feature_df['lateNightDuration']) / \
                                                 risk_feature_df['activeDay']
    risk_feature_df['lateNightDurationPerTrip'] = np.sum(trip_feature_df['lateNightDuration']) / \
                                                  np.sum(trip_feature_df['lateNightFlag'])
    risk_feature_df['lateNightTripRatio'] = np.sum(trip_feature_df['lateNightFlag']) / \
                                            risk_feature_df['tripNum']

    risk_feature_df['highCurveTripRatio'] = np.sum(trip_feature_df['isHighCurveTrip']) / \
                                            risk_feature_df['tripNum']
    risk_feature_df['longDisTripRatio'] = np.sum(trip_feature_df['isLongDistanceTrip']) / \
                                          risk_feature_df['tripNum']
    risk_feature_df['longTripRatio'] = np.sum(trip_feature_df['isLongTimeTrip']) / \
                                       risk_feature_df['tripNum']
    return risk_feature_df


def file_process(file_name):
    print(file_name)
    trip_feature_df = pd.read_csv(file_name)
    trip_feature_df['vin'] = file_name
    risk_feature_df = calculate_risk_features(trip_feature_df)
    return risk_feature_df



def folder_process(basefolder, folderlist, outputfolder):
    risk_feature_dfs = []
    for foldername in folderlist:
        savepath = outputfolder+'/'+foldername+'/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        os.chdir(path=basefolder+foldername)
        for file_name in os.listdir():
            risk_feature_df = file_process(file_name)
            risk_feature_dfs.append(risk_feature_df)
        RISK_FEATURE_DF = pd.concat(risk_feature_dfs)
        RISK_FEATURE_DF.to_csv(savepath+'risk_feature.csv')

folder_process(DATA_FILE, folderlist, outputfolder)
print("finished")

#risk_feature_dfs = []
#for i in trip_data_files:
#    vin = os.path.splitext(os.path.basename(i))[0]
#    trip_feature_df = pd.read_csv(i)
#    trip_feature_df['vin'] = vin
#    risk_feature_df = calculate_risk_features(trip_feature_df)
#    risk_feature_dfs.append(risk_feature_df)

# 2.Concatenate all data into one DataFrame
#RISK_FEATURE_DF = pd.concat(risk_feature_dfs)

# 3.save risk feature data
#RISK_FEATURE_DF.to_csv(RISK_FEATURE_DATA_FILE + "/risk_feature.csv")

