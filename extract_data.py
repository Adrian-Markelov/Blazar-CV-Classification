import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def read_csv(data_source):
    df = pd.read_csv(data_source)
    mag = dict()
    date = dict()
    
    def add_datapoint(dictionary, point):
        dictionary[i] = (dictionary.get(i,[])) + [point]

    for (i, m, d) in zip(df['InputID'], df['Mag'], df['MJD']):
        # print(i, m)
        add_datapoint(mag, m)
        add_datapoint(date, d)
    return (mag, date)

def plot_data(magnitude, date, save_path):
    plt.figure(figsize=(35,40))
    N = 36
    for (i, k) in enumerate(list(magnitude.keys())[:N]):
        date_v = date[k]
        mag_v = magnitude[k]
        date_v -= np.min(date_v)
        plt.subplot(6,6,i+1)
        plt.scatter(date_v, mag_v, 1)
        plt.xlabel('Time')
        plt.ylabel('Mag')
    plt.savefig(save_path)

(mag, date) = read_csv('data/Blazar_LC.csv')
plot_data(mag, date, 'data/Blazar.png')

(mag, date) = read_csv('data/CV_LC.csv')
plot_data(mag, date, 'data/CV.png')

