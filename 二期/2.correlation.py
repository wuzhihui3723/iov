import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

# 1.load data
BASE_FILE = os.path.dirname(os.path.abspath(__file__))  # the directory of the script being run
DATA_FILE = BASE_FILE + '/data'
RESULT_FILE = BASE_FILE + '/result'
RISK_FEATURE_DATA_FILE = DATA_FILE + '/risk_feature'
clean_risk_feature_file = RISK_FEATURE_DATA_FILE + '/clean_risk_feature.csv'
clean_risk_feature_df = pd.read_csv(clean_risk_feature_file)

# 2.calculate correlation
corr = clean_risk_feature_df.corr()
corr.to_csv(RESULT_FILE + '/corr.csv')

# 3.plot corr
corr_plot = sns.heatmap(corr,
                        xticklabels=corr.columns,
                        yticklabels=corr.columns)

plt.figure(figsize=(16, 6))
# corr_plot = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
corr_plot = sns.heatmap(corr)
corr_plot.get_figure().savefig(RESULT_FILE + '/corr_plot.png', dpi=300, bbox_inches='tight')
