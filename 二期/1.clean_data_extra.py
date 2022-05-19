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
data = pd.read_csv(DATA_FILE+'/risk_feature/demo_risk_feature.csv')
# data = pd.read_csv(DATA_FILE+'/risk_feature/500car_risk_feature.csv')

data.loc[data["qdcPhk"] == '#NAME?', "qdcPhk"] = '0'
data["qdcPhk"] = data.qdcPhk.astype('float64')
data.to_csv(DATA_FILE+'/risk_feature/demo_risk_feature_correct.csv')