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

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def get_all_data():
    data_blazar = read_csv('data/Blazar_LC.csv')
    data_cv = read_csv('data/CV_LC.csv')
    data = merge_two_dicts(data_blazar, data_cv)
    return data

if __name__ == '__main__':
    data = get_all_data()
    lengths = []
    for index in data:
        lengths.append(len(data[index]['Mag']))
    print(sorted(lengths))


