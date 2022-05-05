import sys
import os
import datetime
#将原始混杂数据提取到以vin为文件名的各自文件中,结果保存/sdb/output/2020_1001-2000.csv/

def appendFile(filename, content):
    with open(filename,'a') as f:
        f.write(content)


def getKey(lineStr):
    key = lineStr.split('^')[0]
    return key

def processfile(filename, outputfolder):
    timeStart = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(timeStart,"start process",filename,"output",outputfolder,sep='#')
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    with open(filename, 'r') as file:
        for line in file:
            key = line.split('^')[0]
            appendFile(outputfolder+key, line)
    timeEnd = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(timeEnd, "end process", filename, sep='#')



def processFolder(foldername, fileList, outputfolder):
    for fileName in fileList:
        processfile(foldername+fileName, outputfolder+fileName+'/')


fileList = ['2020_1001-2000.csv','2020_2001-3000.csv','2020_4001-5000.csv','2020_5001-6000.csv','2020_6001-7000.csv','2020_7001-8000.csv','2020_8001-9000.csv']
processFolder("/sdb/20201116/", fileList, "/sdb/output/")
print("all finished")