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
            result_dict[current_id][k] = np.array(dataframe[1][k])
    return result_dict

def plot_data(magnitude, date, save_path):
    plt.figure(figsize=(35,40))
    N = 10
    for i in range(len(magnitude)):
        date_v = date[i]
        mag_v = magnitude[i]
        date_v -= np.min(date_v)
        plt.subplot(6,6,i+1)
        plt.scatter(date_v, mag_v, 1)
        plt.xlabel('Time')
        plt.ylabel('Mag')
    plt.savefig(save_path)


def BuildSF(time, mag):
  nr = len(mag)
  allpairs = np.zeros((nr*(nr-1)//2, 2))

  pos = 0
  for i in range(nr):
    for j in range(i+1, nr):
      allpairs[pos,0] = time[i]-time[j]
      allpairs[pos,1] = mag[i]-mag[j]
      pos = pos + 1
  
  # Find the absolute time difference
  timediff = np.abs(allpairs[:,0])
  magdiff = np.log10(np.abs(allpairs[:,1]) + 1e-9)

  bad_idx = np.nonzero(magdiff < -5)[0]
  timediff = np.delete(timediff, bad_idx)
  magdiff = np.delete(magdiff, bad_idx)

  return (timediff,magdiff)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# dictionary is in the form d[index][key]
def get_all_data():
    data_blazar = read_csv('data/Blazar_LC.csv')
    for index in data_blazar:
        data_blazar[index]['class'] = 'Blazar'
    print("We have %d Blazar points" % (len(data_blazar)))

    data_cv = read_csv('data/CV_LC.csv')
    for index in data_cv:
        data_cv[index]['class'] = 'CV'
    print("We have %d CV points" % (len(data_cv)))

    data = dict()
    counter = 0
    for i in data_blazar:
        data[counter] = data_blazar[i]
        counter += 1

    for i in data_cv:
        data[counter] = data_cv[i]
        counter += 1

    return data

# first column is julian date, the next N columns are generated data
def generate_data(data, N):
    new_data = np.zeros((len(data['Mag']), N+1))
    for (idx, time) in enumerate(data['MJD']):
        # import pdb; pdb.set_trace()
        err = data['Magerr'][idx]
        mag = data['Mag'][idx]
        new_data[idx,0] = time
        new_data[idx,1:] = np.random.normal(mag, err**0.5, N)
    return new_data

def plot_structure(x, y):
    plt.figure(1)
    plt.scatter(x, y, s=2)
    plt.show()

# given a structure function output, find the quantiles
def find_quantiles(one_point):
    NUM_QUANTILES = 10



'''

The structure functions are saved in SF.npy as a dictionary A s.t. 
A[idx][0] is timediff, A[idx][1] is magdiff, 0 <= idx <= N 

Extract with the following code:

    # with open('SF.pickle','rb') as F:
    #     import pickle
    #     a = pickle.load(F)

'''

if __name__ == '__main__':

    data = get_all_data()

    # structures = dict()
    # for i in data:
    #     print(i)
    #     (x, y) = BuildSF(data[i]['MJD'], data[i]['Mag'])
    #     structures[i] = dict()
    #     structures[i]['timediff'] = x
    #     structures[i]['magdiff'] = y
    #     structures[i]['class'] = data[i]['class']

    # import pickle
    # with open('SF_bin.pickle','wb') as F:
    #     pickle.dump(structures, F)

    # with open('SF.pickle','w') as F:
    #     pickle.dump(structures, F)










