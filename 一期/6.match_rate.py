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
from sklearn.model_selection import train_test_split

# 1.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))#the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
RISK_FEATURE_RESULT_DATA_FILE = DATA_FILE + '/result'

risk_feature_s_data_files = glob.glob(RISK_FEATURE_RESULT_DATA_FILE + "/s/*/*.csv")
risk_feature_y_data_files = glob.glob(RISK_FEATURE_RESULT_DATA_FILE + "/y/*/*.csv")

risk_feature_s_dfs = []
for i in risk_feature_s_data_files:
    risk_feature_s_df = pd.read_csv(i)
    risk_feature_s_dfs.append(risk_feature_s_df)
risk_feature_s = pd.concat(risk_feature_s_dfs)  # Concatenate all data into one DataFrame
risk_feature_s['vehicle_type'] = 's'

risk_feature_y_dfs = []
for j in risk_feature_y_data_files:
    risk_feature_y_df = pd.read_csv(j)
    risk_feature_y_dfs.append(risk_feature_y_df)
risk_feature_y = pd.concat(risk_feature_y_dfs)  # Concatenate all data into one DataFrame
risk_feature_y['vehicle_type'] = 'y'

# match insurance data
match_rate_s = 0.55
match_rate_y = 0.55
X_no_match_s, X_match_s = train_test_split(risk_feature_s, test_size=match_rate_s, random_state=0)
X_no_match_y, X_match_y = train_test_split(risk_feature_y, test_size=match_rate_y, random_state=0)
risk_feature = pd.concat([X_match_s, X_match_y], axis=0)
risk_feature.to_csv(RISK_FEATURE_DATA_FILE+'/risk_feature.csv', index=False)
