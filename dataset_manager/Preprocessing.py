import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob





def read_merge_data(house, params_application, appliance,path_dataset):

    pos = params_application[appliance]['houses'].index(house)
    path = path_dataset+'low_freq/house_{}/'.format(house)
    file = path + 'channel_{}.dat'.format(params_application[appliance]['main'][pos])
    df = pd.read_table(file, sep=' ', names=['unix_time', "main"],
                       dtype={'unix_time': 'int64', 1: 'float64'})
    print(str(house)+"  "+str(params_application[appliance]['main'][pos]))
    i=params_application[appliance]['channels'][pos]
    file = path + 'channel_{}.dat'.format(i)
    data = pd.read_table(file, sep=' ', names=['unix_time', i],
                         dtype={'unix_time': 'int64', i: 'float64'})
    df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df

def create_house_dataframe(path_dataset,params_application,appliance):

    train = {}
    test = {}
    for i in params_application[appliance]['train_build']:
        #print(i)
        train[i] = read_merge_data(i, params_application, appliance,path_dataset)
        #print("House {} finish:".format(i))
        print(train[i].head())


    t = params_application[appliance]['test_build']
    test[t] = read_merge_data(t, params_application, appliance,path_dataset)
    return train,test
