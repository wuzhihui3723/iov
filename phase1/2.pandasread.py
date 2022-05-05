# import pandas as pd
# #数据探查过程中的交互代码记录
# headerlist = [vin','time','spd','soc','mileage','lon','lat','w2910','w2911','gear','locationStatus','w2901','w2902','w2903','w2904','w2905','w2906','w2907','n2808','c2804','w2909','chargeStatus','brakeStatus','powerStatus','carStatus','status','dcStatus','c2307','w2913','w2914','w2915','w2916','w2917','w2918','w2919','w2930','n2920','c2921','c2923']
#
# data = pd.read_csv('LS5A2DJX1KA003176',sep='^', header=None, names=headerlist)
#
# sortdata = data.sort_values(by='time', ascending=True)
#
# pdata = sortdata[['time','spd','soc','mileage','lon','lat','gear','locationStatus','chargeStatus','brakeStatus','powerStatus','carStatus','status','dcStatus']]
#
# pdata.describe()
#
# #  pdata
# #                    time  spd   soc   mileage          lon         lat  gear  locationStatus  chargeStatus  brakeStatus  powerStatus  carStatus  status  dcStatus
# # 25979    20200107132846  0.0  91.0  122519.0  106504500.0  29529960.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 609188   20200107132856  0.0  91.0  122519.0  106504500.0  29529960.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 238504   20200107132906  0.0  92.0  122519.0  106504500.0  29529960.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 930976   20200107132916  0.0  92.0  122519.0  106504500.0  29529960.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 121201   20200107132926  0.0  93.0  122519.0  106504500.0  29529960.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # ...                 ...  ...   ...       ...          ...         ...   ...             ...           ...          ...          ...        ...     ...       ...
# # 921334   20201116163030  0.0  73.0  936926.0  106584440.0  29645780.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 921310   20201116163040  0.0  73.0  936926.0  106584440.0  29645780.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 1022235  20201116163050  0.0  73.0  936926.0  106584440.0  29645780.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 698695   20201116163100  0.0  74.0  936926.0  106584440.0  29645780.0   0.0             0.0           1.0          0.0          0.0        2.0     1.0       1.0
# # 851782   20201116163102  0.0  74.0  936926.0  106584440.0  29645780.0   0.0             0.0           4.0          0.0          0.0        2.0     1.0       1.0
#
# data.describe()
# #                time           spd           soc       mileage           lon           lat      w2910  ...      w2917      w2918         w2919      w2930         n2920         c2921      c2923
# # count  1.060941e+06  1.059975e+06  1.059975e+06  1.059975e+06  1.059975e+06  1.059975e+06  1059975.0  ...  1059975.0  1059975.0  1.059975e+06  1059975.0  1.059975e+06  1.059975e+06  1059975.0
# # mean   2.020070e+13  1.812278e+02  6.659633e+01  5.789269e+05  1.031945e+08  2.866357e+07        0.0  ...        0.0        0.0  4.820868e-04        0.0  7.606783e-03  4.849171e-04        0.0
# # std    2.795251e+08  1.931798e+02  2.173470e+01  2.381747e+05  1.846721e+07  5.129768e+06        0.0  ...        0.0        0.0  2.195119e-02        0.0  1.230664e-01  2.218625e-02        0.0
# # min    2.020011e+13  0.000000e+00  3.000000e+00  1.225190e+05  0.000000e+00  0.000000e+00        0.0  ...        0.0        0.0  0.000000e+00        0.0  0.000000e+00  0.000000e+00        0.0
# # 25%    2.020052e+13  0.000000e+00  5.100000e+01  4.023210e+05  1.064441e+08  2.953930e+07        0.0  ...        0.0        0.0  0.000000e+00        0.0  0.000000e+00  0.000000e+00        0.0
# # 50%    2.020073e+13  1.280000e+02  7.000000e+01  6.268060e+05  1.064922e+08  2.957724e+07        0.0  ...        0.0        0.0  0.000000e+00        0.0  0.000000e+00  0.000000e+00        0.0
# # 75%    2.020092e+13  3.200000e+02  8.400000e+01  7.719570e+05  1.065385e+08  2.961059e+07        0.0  ...        0.0        0.0  0.000000e+00        0.0  0.000000e+00  0.000000e+00        0.0
# # max    2.020112e+13  1.005000e+03  1.000000e+02  9.369260e+05  1.068002e+08  2.983679e+07        0.0  ...        0.0        0.0  1.000000e+00        0.0  2.000000e+00  2.000000e+00        0.0
#
# #car_500
# pdata['gear'].value_counts()
# # 14.0    812909
# # 0.0     237864
# # 13.0      9202
#
# #13代表 有驱动力：档位编码“1101”， 倒挡
# #14代表 有驱动力：档位编码“1110”， d档
#
# pdata['locationStatus'].value_counts()
# # 0.0    1027083
# # 4.0      32892
# #按照解析4代表 有效定位：北纬：西经，说明有异常点
#
# #状态为4时经纬度均为0
# location = data[data['locationStatus']==4]
# location.describe()
# #                time            spd            soc        mileage       lon       lat           gear  locationStatus      carStatus          n2920
# # count  1.150000e+05  115000.000000  115000.000000  115000.000000  115000.0  115000.0  115000.000000        115000.0  115000.000000  115000.000000
# # mean   2.020072e+13     142.730730      73.961357  539301.113670       0.0       0.0       8.632565             4.0       1.102104       0.011887
# # std    2.855359e+08     195.997503      18.716539  191348.769351       0.0       0.0       6.797261             0.0       0.302787       0.162555
# # min    2.020011e+13       0.000000      10.000000  182808.000000       0.0       0.0       0.000000             4.0       1.000000       0.000000
# # 25%    2.020060e+13       0.000000      61.000000  397220.000000       0.0       0.0       0.000000             4.0       1.000000       0.000000
# # 50%    2.020082e+13       9.000000      77.000000  595181.000000       0.0       0.0      14.000000             4.0       1.000000       0.000000
# # 75%    2.020092e+13     256.000000      90.000000  697055.000000       0.0       0.0      14.000000             4.0       1.000000       0.000000
# # max    2.020112e+13     995.000000     100.000000  827252.000000       0.0       0.0      14.000000             4.0       2.000000       3.000000
#
# pdata['chargeStatus'].value_counts()
# # 3.0    970576
# # 1.0     83004
# # 4.0      6115
# #1 停车充电
# #3 未充电状态
# #4 充电完成
#
# pdata['brakeStatus'].value_counts()
# # 1.0    533784
# # 0.0    526191
# #制动踏板状态
# #0 代表制动关的状态，需要解释，待定
#
# pdata['powerStatus'].value_counts()
# 0.0    596438
# 1.0    463537
# #0 未知，需要解释
# #1 耗电
# #驱动电机状态
#
# pdata['carStatus'].value_counts()
# 1.0    913583
# 2.0    146112
# #1 车辆启动状态
# #2 熄火
#
# pdata['status'].value_counts()
# 1.0    1059695
# #1 纯电
#
# pdata['dcStatus'].value_counts()
# 1.0    1039441
# 2.0      20534
# #1 工作
# #2 断开
#
# #最高报警等级
# data['n2920'].value_counts()
# 0.0    998397
# 2.0      5580
# 3.0       107
# 1.0        15
#
# mileage = sortdata[['time','mileage']]
# mileage['mileage1'] = mileage['mileage'].shift(1)
# mileage['diff']=mileage['mileage']-mileage['mileage1']
# mileage['diff'].value_counts()
# #mileagediff文件
# 跳变和逆序都存在
#
# sta = sortdata[['chargeStatus','carStatus']]
# sta.value_counts()
#
# chargeStatus  carStatus
# 3.0           1.0          880359
# 1.0           2.0           85449
# 3.0           2.0           28236
# 4.0           2.0           10055
# #熄火状态不会处于充电，熄火状态准确
#
# sta = sortdata[['chargeStatus','carStatus','dcStatus']]
# sta.value_counts()
#
# #车辆状态的熄火状态准确,dcStatus字段不使用
# chargeStatus  carStatus  dcStatus
# 3.0           1.0        1.0         880359
# 1.0           2.0        1.0          85449
# 3.0           2.0        1.0          28177
# 4.0           2.0        1.0          10049
# 3.0           2.0        2.0             59
# 4.0           2.0        2.0              6
#
# sortdata['warnsum'] = sortdata['w2901']+sortdata['w2902']+sortdata['w2903']+sortdata['w2904']+sortdata['w2905']+sortdata['w2906']+sortdata['w2907']+sortdata['w2909']+sortdata['w2913']+sortdata['w2914']+sortdata['w2915']+sortdata['w2916']+sortdata['w2917']+sortdata['w2918']+sortdata['w2919']+sortdata['w2930']
# sortdata['warnsum'].value_counts()
# 0.0    1003909
# 1.0        190
#
# warn = sortdata[['n2920','warnsum']]
# warn.value_counts()
# n2920  warnsum
# 0.0    0.0        998397
# 2.0    0.0          5497
# 3.0    1.0           107
# 2.0    1.0            83
# 1.0    0.0            15
#
# gear = sortdata[['gear','carStatus']]
# gear.value_counts()
# gear  carStatus
# 14.0  1.0          707968
# 0.0   1.0          165021
# 0.0   2.0          123740
# 13.0  1.0            7370
# #熄火状态时档位都为0，空挡
#
# sortdata['warn1'] = sortdata['n2920'].shift(1)
# sortdata['warndiff'] = sortdata['n2920']-sortdata['warn1']
# sortdata['warndiff'].value_counts()
#
#  0.0    1003770
# -2.0         10
#  2.0         10
#  1.0          2
# -1.0          2
# -3.0          1
#  3.0          1
#
#  #共计13次报警
#
# #car_6000,第二家企业的车辆，无完全空行上报
#
# pdata['gear'].value_counts()
# 14.0    525372
# 0.0     347178
# 13.0      6850
#
# pdata['locationStatus'].value_counts()
# 0.0    713908
# 4.0    165492
#
# pdata['chargeStatus'].value_counts()
# 3.0    799731
# 1.0     72682
# 4.0      6987
#
# pdata['brakeStatus'].value_counts()
# 1.0    512077
# 0.0    367323
#
# pdata['powerStatus'].value_counts()
# 0.0    552447
# 1.0    326953
#
#  pdata['carStatus'].value_counts()
# 1.0    790995
# 2.0     88405
#
# pdata['status'].value_counts()
# 1.0    879400
#
# pdata['dcStatus'].value_counts()
# 1.0    879378
# 2.0        22
#
# pdata['n2920'].value_counts()
# 0.0    875711
# 2.0      3649
# 1.0        39
# 3.0         1
#
# #location为4时经纬度均为异常值全部为0
# location = pdata[pdata['locationStatus']==4]
# location.describe()
#                time            spd            soc        mileage       lon       lat  ...    brakeStatus    powerStatus      carStatus    status       dcStatus          n2920
# count  1.654920e+05  165492.000000  165492.000000  165492.000000  165492.0  165492.0  ...  165492.000000  165492.000000  165492.000000  165492.0  165492.000000  165492.000000
# mean   2.020072e+13     119.627487      71.385662  297177.358356       0.0       0.0  ...       0.560988       0.275566       1.318076       1.0       1.000060       0.040757
# std    1.336948e+08     189.513391      18.813071   82402.343539       0.0       0.0  ...       0.496268       0.446800       0.465730       0.0       0.007773       0.282445
# min    2.020033e+13       0.000000       0.000000   90492.000000       0.0       0.0  ...       0.000000       0.000000       1.000000       1.0       1.000000       0.000000
# 25%    2.020070e+13       0.000000      60.000000  259818.500000       0.0       0.0  ...       0.000000       0.000000       1.000000       1.0       1.000000       0.000000
# 50%    2.020072e+13       0.000000      75.000000  299869.000000       0.0       0.0  ...       1.000000       0.000000       1.000000       1.0       1.000000       0.000000
# 75%    2.020073e+13     204.000000      86.000000  329693.000000       0.0       0.0  ...       1.000000       1.000000       2.000000       1.0       1.000000       0.000000
# max    2.020112e+13     909.000000     100.000000  525596.000000       0.0       0.0  ...       1.000000       1.000000       2.000000       1.0       2.000000       2.000000
#
#
# #里程数据好很多，但仍存在逆序和跳变
# mileage['diff'].value_counts()
#  0.0        642598
#  2.0        108996
#  1.0        108588
#  3.0         16829
#  4.0          1085
#  5.0           556
#  6.0            91
#  7.0            84
#  8.0            60
#  10.0           32
#  9.0            30
#  15.0           18
#  11.0           15
#  13.0           14
#  12.0           11
#  28.0            6
#  14.0            6
#  17.0            6
#  20.0            5
# -1.0             4
#  18.0            4
#  25.0            4
#  16.0            4
#  22.0            4
#  27.0            3
#  33.0            3
#  24.0            3
#  38.0            2
#  34.0            2
#  35.0            2
#  12203.0         1
#  37.0            1
#  9132.0          1
#  1867.0          1
#  64.0            1
#  658.0           1
#  43.0            1
#  2760.0          1
#  39.0            1
#  21.0            1
#  1490.0          1
#  32.0            1
#  315.0           1
#  19.0            1
#  5707.0          1
#  1019.0          1
#  250.0           1
#  961.0           1
#  1433.0          1
#  1943.0          1
#  40.0            1
#  55.0            1
#  868.0           1
#  853.0           1
#  845.0           1
#  1320.0          1
#  92.0            1
#  23.0            1