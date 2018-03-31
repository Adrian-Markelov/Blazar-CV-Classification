import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def read_csv(data_source):
    df = pd.read_csv(data_source)
    a = list(df.groupby('InputID'))
    result_dict = dict()
    for dataframe in a:
        # dataframe[0] contains inputID
        current_id = dataframe[0]
        result_dict[current_id] = dict()
        for k in dataframe[1]:
            result_dict[current_id][k] = dataframe[1][k]
    return result_dict

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


if __name__ == '__main__':
    data = read_csv('data/Blazar_LC.csv')
    # print(len(mag))
    # plot_data(mag, date, 'data/Blazar.png')

    # (mag, date) = read_csv('data/CV_LC.csv')
    # print(len(mag))
    # plot_data(mag, date, 'data/CV.png')


