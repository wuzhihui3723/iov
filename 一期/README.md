# `ZQY项目`

## `1.该项目包含四部分内容`

`1.1 数据文件夹data，用于存放数据，其中risk_feature文件夹用于存放风险特征数据，trip_feature文件夹用于存在行程特征数据`

`1.2 result文件夹用于存放结果数据`

`1.3 venv文件夹为项目所以来的虚拟环境`

`1.4 项目中的.py文件用于产生结果，并将结果存放于result文件夹中`

## `2.如何使用该项目？`

### `2.0 feature.py`

`执行该文件，该文件读取data/trip_feature中的行程数据文件，生成风险特征因子，并存储与data/risk_feature/risk_feature.csv`

### `2.1 cleanData.py`

`执行执行该文件，`

`首先，该文件读取风险因子数据data/risk_feature/risk_feature.csv`

`其次，计算每个风险因子的1%分位数，并存储为result/quantile.csv`

`然后，清洗数据，生成data/risk_feature/clean_risk_feature.csv`

`最后，对每个风险因子等深分组，可通过bin_num修改分组数量，将生成的分组数据，生成data/risk_feature/bin_risk_feature.csv`

### `2.2 correlation.py`

`执行执行该文件，`

`首先，该文件读取风险因子数据data/risk_feature/clean_risk_feature.csv`

`其次，计算相关系数矩阵，结果保存为result/corr.csv`

`最后，plot相关系数，结果保存为result/corr_plot.png`

### `2.3 create_insurance_data.py`

`执行执行该文件，`

`首先，该文件读取风险因子数据data/risk_feature/clean_risk_feature.csv`

`其次，归一化风险特征，通过weight_df调节每个特征权重，将最终车辆评分保存为result/score.csv`

`最后，生成insurance数据，保存为data/insurance.csv`

### `2.4 oneWayAnalysis.py`

`执行执行该文件，单因子分析` 

### `2.5 GLM.py`

`执行执行该文件，`

`首先，读取分组特征data/risk_feature/bin_risk_feature.csv`

`其次，对训练集建模，共训练两次GLM`

`然后，将第二次GLM模型结果保存在result/params.csv`

`最后，预测测试集，并生成提升曲线数据result/lift_curve.csv，画预测值与真实值提升曲线，并对各变量画出提升曲线`

### `2.6 match_ratio.py`

`匹配保险数据55%`

### `2.7 save_model.py`

`用于预测新的数据集`

`首先，读取新数据 data/risk_feature/data.csv，此数据可更换`

`其次，加载模型，`

`然后，预测新数据，将模型结果保存在result/save_model_prediciton.csv`


