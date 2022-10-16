from calendar import month
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#데이터 정리
csvRead = pd.read_csv('OBS_ASOS_DD_20221008094611.csv', encoding = 'cp949')
csvRead = csvRead[[
    '일시', 
    '평균기온(°C)', 
    '일강수량(mm)', 
    '평균 풍속(m/s)',
    '평균 상대습도(%)',
    '평균 현지기압(hPa)',
    '평균 전운량(1/10)'
    ]]
csvRead = csvRead.rename(columns={
    '일시': 'date', 
    '평균기온(°C)': 'temp', 
    '일강수량(mm)': 'rain', 
    '평균 풍속(m/s)': 'wind', 
    '평균 상대습도(%)': 'humi', 
    '평균 현지기압(hPa)': 'air',
    '평균 전운량(1/10)': 'cloud'
    })
csvRead['date'] = pd.to_datetime(csvRead['date'])
csvRead['year'] = csvRead['date'].dt.year
csvRead['month'] = csvRead['date'].dt.month
csvRead['day'] = csvRead['date'].dt.day
csvRead = csvRead[['year', 'month', 'day', 'temp', 'rain', 'wind', 'humi', 'air', 'cloud']]

#nan(null)처리
csvRead.fillna(0, inplace=True)

# print(csvRead)

train_year = (csvRead['year'] <= 2021) #9년치 학습 데이터
test_year = (csvRead['year'] >= 2022) #1년치 테스트 데이터

def make_data(data):
    x = []
    y = []
    months = list(data['month'])
    temps = list(data['temp'])
    winds = list(data['wind'])
    humis = list(data['humi'])
    airs = list(data['air'])
    clouds = list(data['cloud'])
    rains = list(data['rain'])
    for i in range(len(temps)): #1824번 반복
        temx = []
        temx.extend([
            months[i],
            temps[i],
            winds[i],
            humis[i],
            airs[i],
            clouds[i],
            ])
        x.append(temx)
        y.append(rains[i])
    return (x, y)

train_x, train_y = make_data(csvRead[train_year])
test_x, test_y = make_data(csvRead[test_year])

#학습
lr = LinearRegression(normalize = True)
lr.fit(train_x, train_y) 
pre_y = lr.predict(test_x)

#그래프로 띄워주기
plt.figure(figsize=(10,6), dpi = 100)
plt.plot(test_y, c='r') #실제 값
plt.plot(pre_y, c='b') #예측한 값
plt.ylabel('precipitation(mm)')
plt.xlabel('month')
plt.xticks([0, 32, 60, 91, 121, 152, 182, 213, 244, 274], labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
plt.savefig('10y_lr.png')
plt.show()