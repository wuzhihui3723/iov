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

# 1.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))#the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'

clean_risk_feature_file = RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv'
clean_risk_feature_df = pd.read_csv(clean_risk_feature_file)
clean_risk_feature_df = clean_risk_feature_df.set_index('vin')

insurance_file = DATA_FILE + '/insurance.csv'
insurance_df = pd.read_csv(insurance_file).set_index('vin')

# 2.merge risk features and insurance data
data = clean_risk_feature_df.merge(insurance_df,
                                   how='left',
                                   on='vin')

# # 3. plot_pairplot
# pair_plot = sns.pairplot(data)
# pair_plot.get_figure().savefig(RESULT_FILE + '/pair_plot.png', dpi=300, bbox_inches='tight')

# 3.one way analysis
fig, axs = plt.subplots(nrows=18,
                        ncols=3,
                        figsize=(15, 90))
fig.tight_layout()
# ax = axs.ravel()
for i in clean_risk_feature_df.columns[:-1]:
    axs_loc = clean_risk_feature_df.columns.get_loc(i)
    sns.regplot(x=i, y='y', data=data, ax=axs[axs_loc // 3, axs_loc % 3])
    plt.title("feature : " + i)

# pp = sns.pairplot(data=data,
#                   x_vars=['y'],
#                   y_vars=clean_risk_feature_df.columns)

fig.savefig(RESULT_FILE + '/oneWay_plot.png', dpi=300, bbox_inches='tight')
