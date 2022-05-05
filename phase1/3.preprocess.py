import os
import datetime
import pandas as pd
#将车辆的原始数据去除不需要的字段同时做初步的过滤结果为10个字段,结果保存/sdb/output1/2020_1001-2000.csv/

def processfile(filename, outputfilename):
    timeStart = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(timeStart,"start process",filename,sep='#')

    data = pd.read_csv(filename, sep='^', header=None, usecols=indexlist, names=headerlist,keep_default_na=False)
    data.drop(data[data['locationStatus']==4].index, inplace=True)
    data.drop('locationStatus',axis=1,inplace=True)
    data.drop(data[data['carStatus']==''].index,inplace=True)
    data.drop(data[(data['lat']==0)|(data['lon']==0)].index,inplace=True)
    data.sort_values(by='time', ascending=True, inplace=True)
    data.to_csv(outputfilename, encoding='utf-8', index=0)

    timeEnd = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(timeEnd, "end process", filename, sep='#')



def processFolder(basefolder, folderlist, outputfolder):
    for foldername in folderlist:
        subfoldername = basefolder+foldername+'/'

        outputfoldername = outputfolder+foldername+'/'
        if not os.path.exists(outputfoldername):
            os.makedirs(outputfoldername)

        filenames = os.listdir(subfoldername)
        for filename in filenames:
            processfile(subfoldername+filename, outputfoldername+'/'+filename)


headerlist = ['vin','time','spd','soc','mileage','lon','lat','w2910','w2911','gear','locationStatus','w2901','w2902','w2903','w2904','w2905','w2906','w2907','n2808','c2804','w2909','chargeStatus','brakeStatus','powerStatus','carStatus','status','dcStatus','c2307','w2913','w2914','w2915','w2916','w2917','w2918','w2919','w2930','n2920','c2921','c2923']
indexlist = [1,2,3,4,5,6,9,10,21,24,36]

folderlist = ['2020_1-500']
processFolder("/sdb/test/output/", folderlist, "/sdb/output1/")
print("all finished")