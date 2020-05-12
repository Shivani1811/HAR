from __future__ import print_function
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


#train-phone-accl
columns=['user','activity','time','x','y','z']
data_phone_accel_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/train/phone/accel/'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/train/phone/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_accel_sum=pd.concat([data_phone_accel_sum,temp])
        
        
       
#test-phone-acc
columns=['user','activity','time','x','y','z']

data_phone_accel_test = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/test/phone/accel/'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/test/phone/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_accel_test=pd.concat([data_phone_accel_test,temp])
        
data_phone_accel_sum['z'] = data_phone_accel_sum['z'].str.replace(';','')
data_phone_accel_sum['activity'].value_counts()
data_phone_accel_sum['x']=data_phone_accel_sum['x'].astype('float')
data_phone_accel_sum['y']=data_phone_accel_sum['y'].astype('float')
data_phone_accel_sum['z']=data_phone_accel_sum['z'].astype('float')
data_phone_accel_sum.info()

data_phone_accel_test['z'] = data_phone_accel_test['z'].str.replace(';','')
data_phone_accel_test['activity'].value_counts()
data_phone_accel_test['x']=data_phone_accel_test['x'].astype('float')
data_phone_accel_test['y']=data_phone_accel_test['y'].astype('float')
data_phone_accel_test['z']=data_phone_accel_test['z'].astype('float')
data_phone_accel_test.info()


data_phone_gyro_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/train/phone/gyro'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/train/phone/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_gyro_sum=pd.concat([data_phone_gyro_sum,temp])
        
data_phone_gyro_test = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/test/phone/gyro'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/test/phone/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_phone_gyro_test=pd.concat([data_phone_gyro_test,temp])
        
        
data_phone_gyro_sum['z'] = data_phone_gyro_sum['z'].str.replace(';','')

data_phone_gyro_sum['x']=data_phone_gyro_sum['x'].astype('float')
data_phone_gyro_sum['y']=data_phone_gyro_sum['y'].astype('float')
data_phone_gyro_sum['z']=data_phone_gyro_sum['z'].astype('float')

data_phone_gyro_sum['activity'].value_counts()
data_phone_gyro_sum.info()

data_phone_gyro_test['z'] = data_phone_gyro_test['z'].str.replace(';','')

data_phone_gyro_test['x']=data_phone_gyro_test['x'].astype('float')
data_phone_gyro_test['y']=data_phone_gyro_test['y'].astype('float')
data_phone_gyro_test['z']=data_phone_gyro_test['z'].astype('float')

data_phone_gyro_test['activity'].value_counts()
data_phone_gyro_test.info()



data_watch_gyro_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/train/watch/gyro'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/train/watch/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_gyro_sum=pd.concat([data_watch_gyro_sum,temp])
        
 
data_watch_gyro_test = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/test/watch/gyro'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/test/watch/gyro/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_gyro_test=pd.concat([data_watch_gyro_test,temp])
        
data_watch_gyro_sum['z'] = data_watch_gyro_sum['z'].str.replace(';','')
data_watch_gyro_sum['x']=data_watch_gyro_sum['x'].astype('float')
data_watch_gyro_sum['y']=data_watch_gyro_sum['y'].astype('float')
data_watch_gyro_sum['z']=data_watch_gyro_sum['z'].astype('float')

data_watch_gyro_sum['activity'].value_counts()
data_watch_gyro_sum.info()


data_watch_gyro_test['z'] = data_watch_gyro_test['z'].str.replace(';','')
data_watch_gyro_test['x']=data_watch_gyro_test['x'].astype('float')
data_watch_gyro_test['y']=data_watch_gyro_test['y'].astype('float')
data_watch_gyro_test['z']=data_watch_gyro_test['z'].astype('float')

data_watch_gyro_test['activity'].value_counts()
data_watch_gyro_test.info()


data_watch_accel_sum = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/train/watch/accel'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/train/watch/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_accel_sum=pd.concat([data_watch_accel_sum,temp])
        

data_watch_accel_test = pd.DataFrame(data=None,columns=columns)
for dirname, _, filenames in os.walk('C:/Users/amenda/Desktop/internship/raw/test/watch/accel'):
    for filename in filenames:
        df = pd.read_csv('C:/Users/amenda/Desktop/internship/raw/test/watch/accel/'+filename , sep=",", header=None)
        temp=pd.DataFrame(data=df.values, columns=columns)
        data_watch_accel_test=pd.concat([data_watch_accel_test,temp])
        
 data_watch_accel_sum['z'] = data_watch_accel_sum['z'].str.replace(';','')
data_watch_accel_sum['x']=data_watch_accel_sum['x'].astype('float')
data_watch_accel_sum['y']=data_watch_accel_sum['y'].astype('float')
data_watch_accel_sum['z']=data_watch_accel_sum['z'].astype('float')

data_watch_accel_sum['activity'].value_counts()
data_watch_accel_sum.info()

data_watch_accel_test['z'] = data_watch_accel_test['z'].str.replace(';','')
data_watch_accel_test['x']=data_watch_accel_test['x'].astype('float')
data_watch_accel_test['y']=data_watch_accel_test['y'].astype('float')
data_watch_accel_test['z']=data_watch_accel_test['z'].astype('float')

data_watch_accel_test['activity'].value_counts()
data_watch_accel_test.info()

df_phone = pd.DataFrame(data=None, columns=columns)
df_phone['user']= data_phone_accel_sum['user'].head(1387312)
df_phone['activity']= data_phone_accel_sum['activity'].head(1387312)
df_phone['time']= data_phone_accel_sum['time'].head(1387312)
df_phone['x'] = data_phone_gyro_sum['x'].values + data_phone_accel_sum['x'].head(1387312).values
df_phone['y'] = data_phone_gyro_sum['y'].values + data_phone_accel_sum['y'].head(1387312).values
df_phone['z'] = data_phone_gyro_sum['z'].values + data_phone_accel_sum['z'].head(1387312).values

df_phone_test = pd.DataFrame(data=None, columns=columns)
df_phone_test['user']= data_phone_accel_test['user'].head(986266)
df_phone_test['activity']= data_phone_accel_test['activity'].head(986266)
df_phone_test['time']= data_phone_accel_test['time'].head(986266)
df_phone_test['x'] = data_phone_gyro_test['x'].values + data_phone_accel_test['x'].head(986266).values
df_phone_test['y'] = data_phone_gyro_test['y'].values + data_phone_accel_test['y'].head(986266).values
df_phone_test['z'] = data_phone_gyro_test['z'].values + data_phone_accel_test['z'].head(986266).values

df_watch_test = pd.DataFrame(data=None, columns=columns)
df_watch_test['user']= data_watch_accel_sum['user'].head(997751)
df_watch_test['activity']= data_watch_accel_sum['activity'].head(997751).head(997751)
df_watch_test['time']= data_watch_accel_sum['time'].head(997751)
df_watch_test['x'] = data_watch_gyro_test['x'].values + data_watch_accel_test['x'].head(997751).values
df_watch_test['y'] = data_watch_gyro_test['x'].values + data_watch_accel_test['y'].head(997751).values
df_watch_test['z'] = data_watch_gyro_test['x'].values + data_watch_accel_test['z'].head(997751).values

df_phone_watch = pd.DataFrame(data=None, columns=columns)
df_phone_watch['user']= df_watch['user']
df_phone_watch['activity']= df_watch['activity']
df_phone_watch['time']= df_watch['time']
df_phone_watch['x'] = df_phone['x'].head(1303438).values + df_watch['x'].values
df_phone_watch['y'] = df_phone['y'].head(1303438).values + df_watch['y'].values
df_phone_watch['z'] = df_phone['z'].head(1303438).values + df_watch['z'].values

df_phone_watch_test = pd.DataFrame(data=None, columns=columns)
df_phone_watch_test['user']= df_watch_test['user'].head(986266)
df_phone_watch_test['activity']= df_watch_test['activity'].head(986266)
df_phone_watch_test['time']= df_watch_test['time'].head(986266)
df_phone_watch_test['x'] = df_phone_test['x'].values + df_watch_test['x'].head(986266).values
df_phone_watch_test['y'] = df_phone_test['y'].values + df_watch_test['y'].head(986266).values
df_phone_watch_test['z'] = df_phone_test['z'].values + df_watch_test['z'].head(986266).values

Fs = 20
activities = df_phone_watch['activity'].value_counts().index
df_phone_watch = df_phone_watch.drop(['user', 'time'], axis=1)
df_phone_watch_test = df_phone_watch_test.drop(['user', 'time'], axis=1)
df_a = df_phone_watch[df_phone_watch['activity']=='A'].head(68476)
df_m = df_phone_watch[df_phone_watch['activity']=='M'].head(68476)
df_k = df_phone_watch[df_phone_watch['activity']=='K'].head(68476)
df_p = df_phone_watch[df_phone_watch['activity']=='P'].head(68476)
df_e = df_phone_watch[df_phone_watch['activity']=='E'].head(68476)
df_o = df_phone_watch[df_phone_watch['activity']=='O'].head(68476)
df_c = df_phone_watch[df_phone_watch['activity']=='C'].head(68476)
df_d = df_phone_watch[df_phone_watch['activity']=='D'].head(68476)
df_l = df_phone_watch[df_phone_watch['activity']=='L'].head(68476)
df_s = df_phone_watch[df_phone_watch['activity']=='S'].head(68476)
df_h = df_phone_watch[df_phone_watch['activity']=='H'].head(68476)
df_f = df_phone_watch[df_phone_watch['activity']=='F'].head(68476)
df_g = df_phone_watch[df_phone_watch['activity']=='G'].head(68476)
df_q = df_phone_watch[df_phone_watch['activity']=='Q'].head(68476)
df_r = df_phone_watch[df_phone_watch['activity']=='R'].head(68476)
df_j = df_phone_watch[df_phone_watch['activity']=='J'].head(68476)
df_i = df_phone_watch[df_phone_watch['activity']=='I'].head(68476)
df_b = df_phone_watch[df_phone_watch['activity']=='B']

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([df_a,df_m,df_k,df_p,df_e,df_o,df_c,df_d,df_l,df_b,df_h,df_f,df_g,df_q,df_r,df_s,df_i,df_j]) 

df_a = df_phone_watch_test[df_phone_watch_test['activity']=='A'].head(54058)
df_m = df_phone_watch_test[df_phone_watch_test['activity']=='M'].head(54058)
df_k = df_phone_watch_test[df_phone_watch_test['activity']=='K'].head(54058)
df_p = df_phone_watch_test[df_phone_watch_test['activity']=='P'].head(54058)
df_e = df_phone_watch_test[df_phone_watch_test['activity']=='E'].head(54058)
df_o = df_phone_watch_test[df_phone_watch_test['activity']=='O'].head(54058)
df_c = df_phone_watch_test[df_phone_watch_test['activity']=='C'].head(54058)
df_d = df_phone_watch_test[df_phone_watch_test['activity']=='D'].head(54058)
df_l = df_phone_watch_test[df_phone_watch_test['activity']=='L'].head(54058)
df_b = df_phone_watch_test[df_phone_watch_test['activity']=='B'].head(54058)
df_h = df_phone_watch_test[df_phone_watch_test['activity']=='H'].head(54058)
df_f = df_phone_watch_test[df_phone_watch_test['activity']=='F'].head(54058)
df_g = df_phone_watch_test[df_phone_watch_test['activity']=='G'].head(54058)
df_q = df_phone_watch_test[df_phone_watch_test['activity']=='Q'].head(54058)
df_s = df_phone_watch_test[df_phone_watch_test['activity']=='S'].head(54058)
df_j = df_phone_watch_test[df_phone_watch_test['activity']=='J'].head(54058)
df_i = df_phone_watch_test[df_phone_watch_test['activity']=='I'].head(54058)
df_r = df_phone_watch_test[df_phone_watch_test['activity']=='R']

balanced_data_test = pd.DataFrame()
balanced_data_test = balanced_data_test.append([df_a,df_m,df_k,df_p,df_e,df_o,df_c,df_d,df_l,df_b,df_h,df_f,df_g,df_q,df_r,df_s,df_i,df_j]) 

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity']) 
balanced_data

label = LabelEncoder()
balanced_data_test['label'] = label.fit_transform(balanced_data_test['activity']) 
balanced_data_test

from sklearn.preprocessing import StandardScaler

x = balanced_data[['x','y','z']]
y = balanced_data['label']
scaler = StandardScaler()
x1 = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data=x1, columns=['x','y','z'])
scaled_x['label'] = y.values
scaled_x


xt = balanced_data_test[['x','y','z']]
yt = balanced_data_test['label']
scaler = StandardScaler()
x2 = scaler.fit_transform(xt)
scaled_x_test = pd.DataFrame(data=x2, columns=['x','y','z'])
scaled_x_test['label'] = yt.values
scaled_x_test


import scipy.stats as stats
Fs=20
frame_size = Fs*4 #80
hop_size = Fs*2 #40
def get_frames(df, frame_size, hop_size):
    
    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0,len(df )- frame_size, hop_size):
        x = df['x'].values[i: i+frame_size]
        y = df['y'].values[i: i+frame_size]
        z = df['z'].values[i: i+frame_size]
        
        label = stats.mode(df['label'][i: i+frame_size])[0][0]
        frames.append([x,y,z])
        labels.append(label)
        
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    return frames, labels
    
    
x,y = get_frames(scaled_x, frame_size, hop_size)
xt,yt = get_frames(scaled_x_test, frame_size, hop_size)
x_train = x
y_train = y
x_test = xt
y_test = yt

x_train = x_train.reshape(30813, 80, 3,1)
x_test = x_test.reshape(24325, 80, 3,1)

model = Sequential()
model.add(Conv2D(16, (2,2), activation = 'relu', input_shape = x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2,2), activation = 'relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(18, activation='softmax'))

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
history = model.fit(x_train, y_train, epochs = 200, validation_data=(x_test, y_test), verbose=1 )







