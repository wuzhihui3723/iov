
import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt
import datetime

# 设定车辆原始数据存放路径
#path = '/sdb/output1/2020_1-500/'
#os.chdir(path=path)

# 设定行程切分后保存路径
#save_path = '/sdb/trip/data/2020_1-500/'

# 设定行程特征保存路径
#stat_path = '/sdb/trip/stat/2020_1-500/'
# 读取字段
#initial_key = ['time','spd','soc','mileage','lon','lat','gear','locationStatus','chargeStatus','carStatus','n2920']



# 定义距离计算公式
def distance_delta(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1)/1000000, float(lat1)/1000000, float(lon2)/1000000, float(lat2)/1000000])  # 经纬度转换成弧度
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    return distance
# 定义行驶时长计算公式
def time_delta(starttime,endtime):
    starttime = str(starttime)
    endtime = str(endtime)
    if (datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')-datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')).days == 0:
        delta_seconds = (datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')-datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')).seconds
        delta_days = 0
        act_days = [starttime[:8]]
    else:
        starttime_0 = datetime.datetime.strptime(starttime, '%Y%m%d%H%M%S').replace(hour=0, minute=0, second=0)
        endtime_24 = datetime.datetime.strptime(endtime, '%Y%m%d%H%M%S').replace(hour=23, minute=59, second=59)
        delta_days = (endtime_24 - starttime_0).days
        delta_seconds = (delta_days + 1) * 24 * 3600 - (
                   endtime_24 - datetime.datetime.strptime(endtime, '%Y%m%d%H%M%S')).seconds \
                        - (datetime.datetime.strptime(starttime, '%Y%m%d%H%M%S') - starttime_0).seconds-1
        act_days = []
        for day in range(delta_days + 1):
            act_day = datetime.datetime.strftime(starttime_0 + datetime.timedelta(days=day), '%Y%m%d')
            act_days.append(act_day)
    return delta_seconds,act_days
# 定义午后行驶时长计算
def afternoon_time(starttime,endtime):
    starttime = str(starttime)
    endtime = str(endtime)
    s = datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')
    e = datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')
    if (s.hour>15)|(e.hour<13):
        return 0
    else:
        s_standard = s.replace(hour=13, minute=0, second=0)
        e_standard = e.replace(hour=15, minute=0, second=0)
        if (s < s_standard) & (e >= s_standard)&(e <= e_standard):
            return (e - s_standard).seconds
        elif (s >= s_standard) & (e <= e_standard):
            return (e - s).seconds
        elif (s >= s_standard) & (s <= e_standard) & (e > e_standard):
            return (e_standard - s).seconds
        else:
            return 0
# 定义黄昏行驶时长计算
def dusk_time(starttime,endtime):
    starttime = str(starttime)
    endtime = str(endtime)
    s = datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')
    e = datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')
    if (s.hour>19)|(e.hour<17):
        return 0
    else:
        s_standard = s.replace(hour=17, minute=0, second=0)
        e_standard = e.replace(hour=19, minute=0, second=0)
        if (s < s_standard) & (e >= s_standard)&(e <= e_standard):
            return (e - s_standard).seconds
        elif (s >= s_standard) & (e <= e_standard):
            return (e - s).seconds
        elif (s >= s_standard) & (s <= e_standard) & (e > e_standard):
            return (e_standard - s).seconds
        else:
            return 0
# 定义后半夜行驶时长计算
def latenight_time(starttime,endtime):
    starttime = str(starttime)
    endtime = str(endtime)
    s = datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')
    e = datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')
    s_standard0 = s.replace(hour=0, minute=0, second=0)
    e_standard0 = e.replace(hour=5, minute=0, second=0)
    s_standard1 = s.replace(hour=23, minute=0, second=0)
    e_standard1 = e.replace(hour=23, minute=59, second=59)
    if (s >= s_standard0) & (e <= e_standard0):
        return (e - s).seconds
    elif (s >= s_standard0) & (s <= e_standard0)&(e > e_standard0):
        return (e_standard0 - s).seconds
    elif (s < s_standard1) & (e >= s_standard1) & (e <= e_standard1):
        return (e - s_standard1).seconds
    elif (s >= s_standard1) &  (e <= e_standard1):
        return (e - s).seconds
    else:
        return 0
#################################################################################################################################################################################
# 定义相邻点时长计算
def adjacent_time(starttime,endtime):
    starttime = str(starttime)
    endtime = str(endtime)
    s = datetime.datetime.strptime(starttime,'%Y%m%d%H%M%S')
    e = datetime.datetime.strptime(endtime,'%Y%m%d%H%M%S')
    if starttime[:10] == endtime[:10]:
        return (e-s).seconds
    else:
        e_standard0 = s.replace(hour=23, minute=59, second=59)
        s_standard0 = s.replace(hour=0, minute=0, second=0)
        return (s-e_standard0).seconds + (s_standard0-e).seconds
# 定义电量状态
def soc_status(x,thredhold,status):
    if status == 0:
        if x >= thredhold:
            return 1
        else:
            return 0
    else:
        if x<= thredhold:
            return 1
        else:
            return 0  
# 读取节假日日期数据
holidays = pd.read_csv('holidays.csv',encoding='gbk')
# 定义节假日计算
def holidays_status(x,holiday):
    x_str = str(x)
    if x_str in str(holiday):
        return 1
    else:
        return 0
# 定义周末计算
def weekends(x):
    x_str = str(x)
    x_date = datetime.datetime.strptime(x_str,'%Y%m%d')
    if (x_date.weekday() == 5)|(x_date.weekday() == 6):
        return 1
    else:
        return 0

#################################################################################################################################################################################
def file_process_soc(initial_data, soc_path, file_name):
    initial_data['socStatus'] = initial_data['chargeStatus'].apply(lambda x:1 if x<=2 else 0)
    # 判断状态切换
    initial_data['socStatus_1'] = [initial_data['socStatus'][0]]+list(initial_data['socStatus'][:(len(initial_data)-1)])
    initial_data['trip_point'] = initial_data['socStatus'] - initial_data['socStatus_1']
    start_list = initial_data[initial_data['trip_point'] == 1].index.to_list()
    end_list = initial_data[initial_data['trip_point'] == -1].index.to_list()
    if len(start_list)==0:
        start_list = [0] + start_list
    if len(end_list)==0:
        end_list = end_list + [len(initial_data) - 1]
    # 判断开始结束是否对齐
    if start_list[0] > end_list[0]:
        start_list = [0] + start_list  # 默认第一个点为充电开始
    if start_list[-1] > end_list[-1]:
        end_list = end_list + [len(initial_data) - 1]  # 默认最后一个点为充电结束

    initial_data['soc_id'] = ''

    trip_soc_list = []
    for j in range(len(start_list)):
        data=process_soc(initial_data, start_list, end_list, j)
        if not(data==None ):
            trip_soc_list.append(data)
    if len(trip_soc_list)==0:
	    return
    trip_soc = pd.DataFrame(trip_soc_list, columns=['socId', 'startTime', 'endTime', 'Duration',
                                                    'ChargeDay', 'startSoc', 'endSoc', 'chargeAmount',
                                                    'overChargeFlag','overChargeDur'])
    trip_soc.to_csv(soc_path + '/' + file_name, index=False)


# 定义过充标识计算
def over_charge(x):
    if x>2:
        return 1
    else:
        return 0

def process_soc(initial_data, start_list, end_list, j):
    if (end_list[j]-start_list[j])<2:
        return
    soc_id = 'socid' + '_' + str(j)
#     initial_data.loc[start_list[j]:(end_list[j]+1),'trip_id'] = trip_id
    initial_data.loc[start_list[j]:(end_list[j]+1),'trip_id'] = soc_id

    data_temp = initial_data.loc[start_list[j]:(end_list[j]+1),:].reset_index(drop=True)
    data_temp['time_1'] = [data_temp['time'][0]] + list(data_temp['time'][:(len(data_temp) - 1)])
    data_temp['time_delta'] = data_temp.apply(lambda x:adjacent_time(x['time_1'],x['time']),axis=1)
    data_temp['over_charge'] = data_temp['soc'].apply(lambda x:1 if x == 100 else 0)

    start_time = data_temp['time'][0]
    end_time = data_temp['time'][len(data_temp)-1]
    duration_s,charge_day = time_delta(start_time,end_time)
    start_soc = data_temp['soc'][0]
    end_soc = data_temp['soc'][len(data_temp)-1]
    charge_amount = end_soc - start_soc
    over_charge_flag = over_charge(data_temp['over_charge'].sum())
    over_charge_dur = data_temp.loc[data_temp['over_charge'] == 1,'time_delta'].sum()
    
    data = [soc_id,start_time,end_time,duration_s,charge_day,start_soc,end_soc,charge_amount,over_charge_flag,over_charge_dur]
    return data


def file_process_trip(initial_data, stat_path, file_name):
    initial_data['carStatus'] = initial_data['carStatus'].apply(lambda x:1 if x == 1 else 2)
    initial_data['carStatus_1'] = [initial_data['carStatus'][0]]+list(initial_data['carStatus'][:(len(initial_data)-1)])
    initial_data['trip_point'] = initial_data['carStatus'] - initial_data['carStatus_1']
    start_list = initial_data[initial_data['trip_point'] == -1].index.to_list()
    end_list = initial_data[initial_data['trip_point'] == 1].index.to_list()
    if len(start_list)==0:
        start_list = [0] + start_list
    if len(end_list)==0:
        end_list = end_list + [len(initial_data) - 1]
    # 判断开始结束是否对齐
    if start_list[0] > end_list[0]:
        start_list = [0] + start_list  # 默认第一个点为行程开始
    if start_list[-1] > end_list[-1]:
        end_list = end_list + [len(initial_data) - 1]  # 默认最后一个点为结束

    initial_data['trip_id'] = ''

    trip_stat_list = []
    for j in range(len(start_list)):
        data=process_trip(initial_data, start_list, end_list, j)
        if not(data==None ):
            trip_stat_list.append(data)
#################################################################################################################################################################################
    #print(trip_stat_list)
    if len(trip_stat_list)==0:
        return
    trip_stat =pd.DataFrame(trip_stat_list, columns = ['tripId','distance','startLat','startLon',
                             'startTime','endLat','endLon','endTime',
                             'duration','activeDay','TripCurve',
                             'isLongDistanceTrip','isLongTimeTrip',
                             'isHighCurveTrip','avgSpd','afternoonDis',
                             'afternoonDuration','afternoonFlag',
                             'duskDis','duskDuration','duskFlag',
                             'lateNightDis','lateNightDuration',
                             'lateNightFlag','brakeCount','warnCount',
                             'bdCount',
                              'lowSpdDuration','maxSpd','startOdo','endOdo',
                              'odo','odoErrorCount','gpsDistance','startSoc',
                              'endSoc','useSoc','isLowSocTrip','isFullSocTrip',
                              'isHolidayTtrip','isWeekendTrip'])
#################################################################################################################################################################################
    trip_stat.to_csv(stat_path + '/' + file_name, index=False)



def time_ab(x):
    try:
        x = str(x)
        x = datetime.datetime.strptime(x,'%Y%m%d%H%M%S')
        return 1
    except:
        return 0

#initial_key = ['time','spd','soc','mileage','lon','lat','gear','locationStatus','chargeStatus','carStatus','n2920']
#file_name为vin
def file_process(file_name,stat_path,soc_path):
    print(file_name)

    initial_data = pd.read_csv(file_name, sep='^', header=None, names=headerlist, dtype={'time':object})
    
    if len(initial_data.index) < 2:
        return

    initial_data.dropna(inplace=True)
    initial_data.drop(initial_data[initial_data['locationStatus']==4].index, inplace=True)
    initial_data.drop('locationStatus',axis=1,inplace=True)
    initial_data.drop(initial_data[(initial_data['carStatus']=='')|(initial_data['lon']==0)].index,inplace=True)
    initial_data.sort_values(by='time', ascending=True, inplace=True)
    initial_data.drop(initial_data[(initial_data['lat']<1000000)|(initial_data['lon']<1000000)|(initial_data['lat']>90000000)|(initial_data['lon']>180000000)].index,inplace=True)
    initial_data['time_ab'] = initial_data['time'].apply(lambda x:time_ab(x))
    initial_data.drop(initial_data[initial_data['time_ab']==0].index, inplace=True)
    initial_data.reset_index(drop=True,inplace=True)
    
    if len(initial_data.index) < 2:
        return
    # 判断状态切换

    file_process_trip(initial_data, stat_path, file_name)

    file_process_soc(initial_data, soc_path, file_name)
    

def process_trip(initial_data, start_list, end_list, j):
    if (end_list[j]-start_list[j])<2:
        return
    trip_id = 'tripid' + '_' + str(j)
    initial_data.loc[start_list[j]:(end_list[j]+1),'trip_id'] = trip_id
    data_temp = initial_data.loc[start_list[j]:(end_list[j]+1),:].reset_index(drop=True)
    data_temp['lon_1'] = [data_temp['lon'][0]] + list(data_temp['lon'][:(len(data_temp) - 1)])
    data_temp['lat_1'] = [data_temp['lat'][0]] + list(data_temp['lat'][:(len(data_temp) - 1)])
    data_temp['distance_delta'] = data_temp.apply(lambda x:distance_delta(x['lon_1'],x['lat_1'],x['lon'],x['lat']),axis=1)
    data_temp['time_1'] = [data_temp['time'][0]] + list(data_temp['time'][:(len(data_temp) - 1)])
    #############################################################################################################################################################################
    data_temp['time_delta'] = data_temp.apply(lambda x:adjacent_time(x['time_1'],x['time']),axis=1)
    data_temp['mileage_1'] = [data_temp['mileage'][0]] + list(data_temp['mileage'][:(len(data_temp) - 1)])
    data_temp['mileage_ab'] = data_temp.apply(lambda x:1 if x['mileage'] < x['mileage_1'] else 0,axis=1)
    #############################################################################################################################################################################
    data_temp['afternoon_duration'] = data_temp.apply(lambda x:afternoon_time(x['time_1'],x['time']),axis=1)
    data_temp['dusk_duration'] = data_temp.apply(lambda x: dusk_time(x['time_1'], x['time']), axis=1)
    data_temp['latenight_duration'] = data_temp.apply(lambda x: latenight_time(x['time_1'], x['time']), axis=1)

    distance_m = data_temp['distance_delta'].sum()
    start_lat = data_temp['lat'][0]
    start_lon = data_temp['lon'][0]
    start_time = data_temp['time'][0]
    end_lat = data_temp['lat'][len(data_temp)-1]
    end_lon = data_temp['lon'][len(data_temp)-1]
    end_time = data_temp['time'][len(data_temp)-1]
    duration_s,active_day = time_delta(start_time,end_time)
    distance_between=distance_delta(data_temp['lon'][len(data_temp)-1],data_temp['lat'][len(data_temp)-1],data_temp['lon'][0],data_temp['lat'][0])
    if distance_between == 0:
        trip_curve = 0
    else:
        trip_curve = distance_m/distance_between
    
    if distance_m >= 200000: # 200公里
        is_long_distance_trip = 1
    else:
        is_long_distance_trip = 0
    if duration_s > 7200:
        is_long_time_trip = 1
    else:
        is_long_time_trip = 0
    if trip_curve > 2:
        is_high_curve_trip = 1
    else:
        is_high_curve_trip = 0
    if duration_s>0:
        avg_spd = distance_m/duration_s
    else:
        avg_spd = 0
    afternoon_distance = data_temp.loc[data_temp['afternoon_duration']>0,'distance_delta'].sum()
    afternoon_duration = data_temp['afternoon_duration'].sum()
    if afternoon_duration > 0:
        afternoon_flag = 1
    else:
        afternoon_flag = 0
    dusk_distance = data_temp.loc[data_temp['dusk_duration']>0,'distance_delta'].sum()
    dusk_duration = data_temp['dusk_duration'].sum()
    if dusk_duration > 0:
        dusk_flag = 1
    else:
        dusk_flag = 0
    latenight_distance = data_temp.loc[data_temp['latenight_duration']>0,'distance_delta'].sum()
    latenight_duration = data_temp['latenight_duration'].sum()
    if latenight_duration > 0:
        latenight_flag = 1
    else:
        latenight_flag = 0
    brake_count = len(data_temp[data_temp['n2920']>2])
    warn_count = len(data_temp[data_temp['n2920']>1])
    bd_count = len(data_temp[data_temp['n2920']>0])
    #############################################################################################################################################################################
    lowspdduration = data_temp.loc[data_temp['spd'] < 20, 'time_delta'].sum()
    maxspd = data_temp['spd'].max()
    startodo = data_temp['mileage'][0]
    endodo = data_temp['mileage'][len(data_temp)-1]
    odo = endodo - startodo
    odoerrorcount = data_temp['mileage_ab'].sum()
    gpsdistance = distance_between
    startsoc = data_temp['soc'][0]
    endsoc = data_temp['soc'][len(data_temp)-1]
    usesco = endsoc - startsoc
    islowsoctrip = soc_status(data_temp['soc'][0], 20, 1)  # 此处低电量为20
    isfullsoctrip = soc_status(data_temp['soc'][0], 100, 0)
    isholidaytrip = holidays_status(active_day[0],holidays['holiday'])
    isweekendtrip = weekends(active_day[0])
    #############################################################################################################################################################################

    data = [trip_id,distance_m,start_lat,start_lon,start_time,end_lat,end_lon,end_time,duration_s,
    active_day,trip_curve,is_long_distance_trip,is_long_time_trip,is_high_curve_trip,
    avg_spd,afternoon_distance,afternoon_duration,afternoon_flag,
    dusk_distance,dusk_duration,dusk_flag,latenight_distance,latenight_duration,latenight_flag,
    brake_count,warn_count,bd_count,
    lowspdduration,maxspd,startodo,endodo,odo,odoerrorcount,gpsdistance,startsoc,endsoc,usesco,
    islowsoctrip,isfullsoctrip,isholidaytrip,isweekendtrip]
    return data


def folder_process(basefolder, folderlist, outputfolder):
    for foldername in folderlist:
        statpath = outputfolder+'/stat/'+foldername
        socpath = outputfolder+'/soc/'+foldername
        if not os.path.exists(statpath):
            os.makedirs(statpath)
        if not os.path.exists(socpath):
            os.makedirs(socpath)

        os.chdir(path=basefolder+foldername)
        for file_name in os.listdir():
            file_process(file_name, statpath, socpath)


headerlist = ['vin','time','spd','soc','mileage','lon','lat','gear','locationStatus','chargeStatus','carStatus','n2920']
# 循环遍历车辆原始数据
basefolder = '/sdb/output1/'
folderlist = ['2020_1001-2000.csv','2020_2001-3000.csv','2020_3001-4000.csv','2020_4001-5000.csv','2020_5001-6000.csv','2020_501-1000','2020_6001-7000.csv','2020_7001-8000.csv','2020_8001-9000.csv']
outputfolder = '/sdb/trip/'
folder_process(basefolder, folderlist, outputfolder)
print('all finished')
