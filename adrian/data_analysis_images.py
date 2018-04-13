IP.enable_gui = lambda x: False
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pickle; import numpy as np

#%matplotlib inline






#0-217 blazar, 218-793 CV

# scale the data to be 3 x 3
def scale_to_square(_diff):
    _diff = _diff - np.min(_diff) # Shift lower bound to zero
    _diff = 3.0 * (_diff / np.max(_diff)) # Normalize with max and scale by 3 
    return _diff

# Take the SF, idx (index of COSMO Object) and the size of image to generate
# Then return a (im_size, im_size) image of the structure function
def make_one_picture(sf, idx, im_size):
    # Retrieve Structure function for idx index
    (timediff, magdiff, obj_class) = sf[idx]['timediff'], sf[idx]['magdiff'], sf[idx]['class']

    # Scale the SF to be of domain and range of (3.0,3.0)
    scaled_tdiff = scale_to_square(timediff)
    scaled_mdiff = scale_to_square(magdiff)
    
    NUM_BINS = im_size
    H, xedges, yedges = np.histogram2d(scaled_tdiff, scaled_mdiff, bins=NUM_BINS, normed=True)
    
    #link the class as a binary value
    c = 0 if obj_class == 'Blazar' else 1
    
    return np.array(H).T, c
    
    

# PLOTTING
#width = 15
#plt.figure(figsize=(width,width*2/3)) 
#plt.scatter(scaled_tdiff, scaled_mdiff, marker = '.')

def plot_sf_image(image, im_size):
    fig = plt.figure(figsize=(im_size, im_size))
    plt.imshow(image, interpolation='nearest', origin='low')
    plt.colorbar()





def image_2_vect(image):
    lin_image = np.zeros((image.shape[0]**2,))
    for i in range(0,image.shape[0]):
        for j in range(0, image.shape[0]):
            lin_image[i*image.shape[0]+j] = image[i,j]
    return lin_image




def save_raw_images(file, sf, im_size):
    n = len(sf)
    data_images = {}
    for obj_idx in range(0, n):
        (I,c) = make_one_picture(sf,obj_idx, im_size)
        # Save dictionary of images paired with class (save to folder specifically for im_size images)
        sf_image = {'image': I, 'class': c}
        data_images.update({obj_idx: sf_image})
    
    
    with open(file, 'wb') as handle:
        pickle.dump(data_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def get_raw_images(file):
    return pickle.load(open(file, "rb"))


# Saves a data variable
# data = (data_image_vects, data_image_vects_classes)
# data_image_vects = np.array([iv1,iv2,...,ivn])
# data_image_vects_classes = np.array([c1,c2,...,cn])
def save_raw_image_vects(file, sf_images):
    
    n = len(sf_images)
    im_size = sf_images[0]['image'].shape[0]
    data_image_vects = np.zeros((n,im_size**2))
    data_image_vects_classes = np.zeros((n,im_size**2))
    for obj_idx in range(0,n):
        image = sf_images[obj_idx]['image']
        c = sf_images[obj_idx]['class']
        image_vect = image_2_vect(image)
        data_image_vects[obj_idx,:] = image_vect
        data_image_vects_classes[obj_idx] = c
        
    data = (data_image_vects, data_image_vects_classes)
    
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_raw_image_vects(file):
    return pickle.load(open(file,"rb"))
    
    




#sf = pickle.load(open("../data/SF.pickle", "rb"))


images_file = "../data/100p/data_images.pickle"
sf_images = get_raw_images(images_file)




obj_idx = 604
im_size = 100

#image = sf_images[obj_idx]['image']
#plot_sf_image(image, im_size)


image_vects_file = "../data/100p_vect/data_image_vects.pickle"


(sf_image_vects, sf_image_vects_classes) = get_raw_image_vects(image_vects_file)



# calculating Eigenvectors
# Standardize the data
from sklearn.preprocessing import StandardScaler
X = sf_image_vects
X_std = StandardScaler().fit_transform(X)

# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance





abs_cum_var_exp = np.abs(cum_var_exp)
abs_var_exp = np.abs(var_exp)

trace1 = go.Scatter(
    x=list(range(im_size**2)),
    y= abs_cum_var_exp,
    mode='lines+markers',
    name="'Cumulative Explained Variance'",
    hoverinfo= abs_cum_var_exp,
    line=dict(
        shape='spline',
        color = 'goldenrod'
    )
)


trace2 = go.Scatter(
    x=list(range(im_size**2)),
    y= abs_var_exp,
    mode='lines+markers',
    name="'Individual Explained Variance'",
    hoverinfo = abs_var_exp,
    line=dict(
        shape='linear',
        color = 'black'
    )
)
fig = tls.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.5}],
                          print_grid=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2,1,1)
fig.layout.title = 'Explained Variance plots - Full and Zoomed-in'
fig.layout.xaxis = dict(range=[0, 80], title = 'Feature columns')
fig.layout.yaxis = dict(range=[0, 60], title = 'Explained Variance')
fig['data'] += [go.Scatter(x= list(range(im_size**2)) , y=abs_cum_var_exp, xaxis='x2', yaxis='y2', name = 'Cumulative Explained Variance')]
fig['data'] += [go.Scatter(x=list(range(im_size**2)), y=abs_var_exp, xaxis='x2', yaxis='y2',name = 'Individual Explained Variance')]

# fig['data'] = data
# fig['layout'] = layout
# fig['data'] += data2
# fig['layout'] += layout2
py.iplot(fig, filename='inset example')







