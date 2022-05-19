import pandas as pd
import numpy as np
import datetime

import os
import sys
import pathlib
import glob
import ast
from math import sin, cos, sqrt, atan2, radians
from sklearn.neighbors import DistanceMetric

# #0.create trip features
# trip_feature_df = pd.DataFrame({'vin': ['vin10'] * 6,
#                                 'trip_id': ['trip1', 'trip2', 'trip3', 'trip4','trip5','trip6'],
#                                 'distance': np.random.uniform(0, 1, 6)*1000,
#                                 'duration': np.random.uniform(0, 1, 6)*100,
#                                 'TripCurve': np.random.uniform(0, 1, 6),
#                                 'isLongDistanceTrip': np.random.randint(0, 2, 6),
#                                 'isLongTimeTrip': np.random.randint(0, 2, 6),
#                                 'isHighCurveTrip': np.random.randint(0, 2, 6),
#                                 'activeDay': [['20110908','20110909'], ['20110909'], ['20110910']]*2,
#                                 'brakeCount': np.random.randint(0, 5, 6),
#                                 'warnCount': np.random.randint(0, 5, 6),
#                                 'bdCount': np.random.randint(0, 5, 6),
#                                 'afternoonDis': np.random.uniform(0, 1, 6)*1000,
#                                 'afternoonDuration': np.random.uniform(0, 120, 6),
#                                 'afternoonFlag': np.random.randint(0, 2, 6),
#                                 'duskDis': np.random.uniform(0, 1, 6)*1000,
#                                 'duskDuration': np.random.uniform(0, 120, 6),
#                                 'duskFlag': np.random.randint(0, 2, 6),
#                                 'endLat': np.random.uniform(0, 20, 6),
#                                 'endLon': np.random.uniform(0, 20, 6),
#                                 'endTime': [np.random.choice(pd.date_range(datetime.datetime(2013,1,1),datetime.datetime(2013,1,3))) for i in range(6)],
#                                 'lateNightDis': np.random.uniform(0, 1, 6)*1000,
#                                 'lateNightDuration': np.random.uniform(0, 120, 6),
#                                 'lateNightFlag': np.random.uniform(0, 2, 6),
#                                 'startLat': np.random.uniform(0, 20, 6),
#                                 'startLon': np.random.uniform(0, 20, 6),
#                                 'startTime': [np.random.choice(pd.date_range(datetime.datetime(2013,1,1),datetime.datetime(2013,1,3))) for i in range(6)]
#                                 })
# trip_feature_df.to_csv('data/trip_feature/trip_sample9.csv')

#1.load data
#BASE_FILE = os.path.dirname(os.path.abspath(__file__))#the directory of the script being run
#DATA_FILE = BASE_FILE + '/data'
#TRIP_FEATURE_DATA_FILE = DATA_FILE + '/trip_feature'
#RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
#CHARGE_FEATURE_DATA_FILE = DATA_FILE + '/charge_feature'

#trip_data_files = glob.glob(TRIP_FEATURE_DATA_FILE + "/*.csv")
#charge_data_files = glob.glob(CHARGE_FEATURE_DATA_FILE + "/*.csv")


# calculate risk features
def calculate_risk_features(trip_feature_df, charge_feature_df):
    """
        Helper function for calculate_risk_features
        Adjust input argument to the specified origin
        Parameters
        ----------
        trip_feature_df : pandas dataframe
            store trip features
        charge_feature_df : pandas dataframe
            store charge features

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

    risk_feature_df['warnCountPhk'] = risk_feature_df['warnCountPhk'] / (risk_feature_df['distance']/1000)*100
    risk_feature_df['bdPhk'] = risk_feature_df['bdPhk'] / (risk_feature_df['distance'] / 1000) * 100

    risk_feature_df['avgSpd'] = risk_feature_df['distance'] / risk_feature_df['duration']
    risk_feature_df['disPerTrip'] = risk_feature_df['distance'] / risk_feature_df['tripNum']
    risk_feature_df['activeDay'] = len(set(trip_feature_df['activeDay'].sum()))
    # activeDayRatio
    trip_feature_df['startDate'] = pd.to_datetime(trip_feature_df['startTime'].astype(str),
                                                  format='%Y%m%d%H%M%S')
    trip_feature_df['endDate'] = pd.to_datetime(trip_feature_df['endTime'].astype(str),
                                                format='%Y%m%d%H%M%S')
    observation_duration = (trip_feature_df['endDate'][len(trip_feature_df)-1] -
                            trip_feature_df['startDate'][0]).days
    risk_feature_df['activeDayRatio'] = risk_feature_df['activeDay']/observation_duration

    # mileage
    risk_feature_df['mileage'] = risk_feature_df['distance'] / observation_duration * 365

    risk_feature_df['distancePerDay'] = risk_feature_df['distance'] / risk_feature_df['activeDay']
    risk_feature_df['durationPerDay'] = risk_feature_df['duration'] / risk_feature_df['activeDay']
    risk_feature_df['tripNumPerDay'] = risk_feature_df['tripNum'] / risk_feature_df['activeDay']
    # risk_feature_df['idleDurationPerDay'] = np.sum(trip_feature_df['idleDuration']) /\
    #                                         risk_feature_df['activeDay']
    risk_feature_df['lowSpdDurationPerDay'] = np.sum(trip_feature_df['lowSpdDuration']) / \
                                            risk_feature_df['activeDay']
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
    risk_feature_df['holidayTripRatio'] = np.sum(trip_feature_df['isHolidayTrip']) / \
                                          risk_feature_df['tripNum']
    risk_feature_df['weekendTripRatio'] = np.sum(trip_feature_df['isWeekendTrip']) /\
                                          risk_feature_df['tripNum']

    # mainRoute and tripDistEntropy
    trip_feature_df['startLat2'] = round(trip_feature_df['startLat'], 2)
    trip_feature_df['startLon2'] = round(trip_feature_df['startLon'], 2)
    trip_feature_df['endLat2'] = round(trip_feature_df['endLat'], 2)
    trip_feature_df['endLon2'] = round(trip_feature_df['endLon'], 2)
    Route = trip_feature_df.groupby(['startLat2', 'startLon2', 'endLat2', 'endLon2']).\
        agg({'tripId': 'count'}).sort_values('tripId', ascending=False)
    Route['tripRatio'] = Route['tripId']/np.sum(Route['tripId'])
    if len(Route) >= 3:
        risk_feature_df['1stRouteTripRatio'] = Route['tripRatio'].iloc[0]
        risk_feature_df['2ndRouteTripRatio'] = Route['tripRatio'].iloc[1]
        risk_feature_df['3rdRouteTripRatio'] = Route['tripRatio'].iloc[2]
    elif len(Route) == 2 :
        risk_feature_df['1stRouteTripRatio'] = Route['tripRatio'].iloc[0]
        risk_feature_df['2ndRouteTripRatio'] = Route['tripRatio'].iloc[1]
        risk_feature_df['3rdRouteTripRatio'] = 0
    else:
        risk_feature_df['1stRouteTripRatio'] = Route['tripRatio'].iloc[0]
        risk_feature_df['2ndRouteTripRatio'] = 0
        risk_feature_df['3rdRouteTripRatio'] = 0

    risk_feature_df['tripDistEntropy'] = -np.sum(np.log(Route['tripRatio']) * Route['tripRatio'])

    # tripMissCount & missMileage
    dfr = trip_feature_df.copy()
    dfr.startLat = np.radians(trip_feature_df['startLat'])
    dfr.endLat = np.radians(trip_feature_df['endLat'])
    dfr.startLon = np.radians(trip_feature_df['startLon'])
    dfr.endLon = np.radians(trip_feature_df['endLon'])
    dfr_start = dfr[['startLat', 'startLon']]
    dfr_end = dfr[['endLat', 'endLon']]
    hs = DistanceMetric.get_metric("haversine")
    R = 6371  # radius of earth
    distance = (hs.pairwise(dfr_start, dfr_end) * R)  # Earth radius in km
    trip_feature_df['missDistance'] = np.append(np.array(0), distance.diagonal(-1))
    trip_feature_df['isMissTrip'] = (trip_feature_df['missDistance'] > 10) & \
                                    (trip_feature_df['gpsDistance'] > 5)
    risk_feature_df['tripMissCount'] = np.sum(trip_feature_df['isMissTrip'])
    risk_feature_df['missMileage'] = np.sum(trip_feature_df['isMissTrip'] *
                                            trip_feature_df['missDistance'])

    # 新能源特征
    trip_feature_df['useSco'] = trip_feature_df['startSoc'] - trip_feature_df['endSoc']
    risk_feature_df['qdcPerTrip'] = np.sum(trip_feature_df['useSco'])/risk_feature_df['tripNum']
    risk_feature_df['qdcPhk'] = np.sum(trip_feature_df['useSco']) / (risk_feature_df['distance'] / 1000) * 100
    risk_feature_df['startSoc'] = np.mean(trip_feature_df['startSoc'])
    risk_feature_df['endSoc'] = np.mean(trip_feature_df['endSoc'])
    risk_feature_df['lowQTripNum'] = np.sum(trip_feature_df['isLowSocTrip'])
    risk_feature_df['fullQTripNum'] = np.sum(trip_feature_df['isFullSocTrip'])

    risk_feature_df['chargeNum'] = charge_feature_df.shape[0]
    risk_feature_df['chargeStartSoc'] = np.mean(charge_feature_df['startSoc'])
    risk_feature_df['chargeEndSoc'] = np.mean(charge_feature_df['endSoc'])
    # charge_feature_df['startDate'] = pd.to_datetime(charge_feature_df['startTime'].astype(str),
    #                                                 format='%Y%m%d%H%M%S')
    # charge_feature_df['endDate'] = pd.to_datetime(charge_feature_df['endTime'].astype(str),
    #                                               format='%Y%m%d%H%M%S')
    # charge_feature_df['duration'] = (charge_feature_df['endDate'] - charge_feature_df['startDate']) / \
    #                                 pd.Timedelta(hours=1)
    risk_feature_df['avgDurPerCharge'] = np.sum(charge_feature_df['Duration'])/risk_feature_df['chargeNum']
    risk_feature_df['overChargeTime'] = np.sum(charge_feature_df['overChargeFlag'])
    risk_feature_df['overChargeDur'] = np.sum(charge_feature_df['Duration']*charge_feature_df['overChargeFlag'])

    # chargeDayNum
    if (charge_feature_df['ChargeDay'] == 0).any():
        risk_feature_df['ChargeDayNum'] = 0
    else:
        charge_feature_df['ChargeDay'] = charge_feature_df['ChargeDay']. \
            apply(lambda x: ast.literal_eval(str(x)))
        risk_feature_df['ChargeDayNum'] = len(set(charge_feature_df['ChargeDay'].sum()))


    # qcPerCharge
    risk_feature_df['qcPerCharge'] = np.sum(charge_feature_df['startSoc'] * charge_feature_df['chargeAmount']) / \
                                     risk_feature_df['chargeNum']

    return risk_feature_df

    
    
    
def file_process(file_name, foldername):
    print(file_name)
    trip_feature_df = pd.read_csv(file_name)
    trip_feature_df['vin'] = file_name
    # 纠正字段名
    trip_feature_df = trip_feature_df.rename(columns={"isHolidayTtrip": "isHolidayTrip"})
    try:
        charge_data_file = CHARGE_FEATURE_DATA_FILE + foldername+'/'+file_name
        charge_feature_df = pd.read_csv(charge_data_file)
    except OSError as e:
        charge_feature_df = pd.DataFrame({'socId': ['socid_0'],
                                          'startTime': [0],
                                          'endTime': [0],
                                          'Duration': [0],
                                          'ChargeDay': [0],
                                          'startSoc': [0],
                                          'endSoc': [0],
                                          'chargeAmount': [0],
                                          'overChargeFlag': [0],
                                          'overChargeDur': [0],
                                          })
    risk_feature_df = calculate_risk_features(trip_feature_df, charge_feature_df)
    return risk_feature_df



def folder_process(folderlist, outputfolder):
    for foldername in folderlist:
        risk_feature_dfs = []
        print('processfolder#'+foldername)
        savepath = outputfolder+'/'+foldername+'/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        os.chdir(path=TRIP_FEATURE_DATA_FILE+foldername)
        for file_name in os.listdir():
            risk_feature_df = file_process(file_name, foldername)
            risk_feature_dfs.append(risk_feature_df)
        RISK_FEATURE_DF = pd.concat(risk_feature_dfs)
        RISK_FEATURE_DF.to_csv(savepath+'risk_feature.csv')
        
        

DATA_FILE = '/sdb/trip/'
TRIP_FEATURE_DATA_FILE = DATA_FILE + '/stat/'
CHARGE_FEATURE_DATA_FILE = DATA_FILE + '/soc/'
outputfolder = '/sdb/risk/'
folderlist = ['y2019_5001-10000.csv']
folder_process(folderlist, outputfolder)
print("finished")


#risk_feature_dfs = []
#for i in trip_data_files:
#    vin = os.path.splitext(os.path.basename(i))[0]
#    trip_feature_df = pd.read_csv(i)
#    trip_feature_df['vin'] = vin
#    # 纠正字段名
#    trip_feature_df = trip_feature_df.rename(columns={"isHolidayTtrip": "isHolidayTrip"})
#    try:
#        charge_data_file = CHARGE_FEATURE_DATA_FILE + '/' + vin + '.csv'
#        charge_feature_df = pd.read_csv(charge_data_file)
#        risk_feature_df = calculate_risk_features(trip_feature_df, charge_feature_df)
#    except OSError as e:
#        charge_feature_df = pd.DataFrame({'socId': ['socid_0'],
#                                          'startTime': [0],
#                                          'endTime': [0],
#                                          'Duration': [0],
#                                          'ChargeDay': [0],
#                                          'startSoc': [0],
#                                          'endSoc': [0],
#                                          'chargeAmount': [0],
#                                          'overChargeFlag': [0],
#                                          'overChargeDur': [0],
#                                          })
#        risk_feature_df = calculate_risk_features(trip_feature_df, charge_feature_df)
#    risk_feature_dfs.append(risk_feature_df)

# 2.Concatenate all data into one DataFrame
#RISK_FEATURE_DF = pd.concat(risk_feature_dfs)

# 3.save risk feature data
#RISK_FEATURE_DF.to_csv(RISK_FEATURE_DATA_FILE + "/risk_feature.csv")

