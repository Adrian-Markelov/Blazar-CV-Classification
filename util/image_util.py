
# Basic Plotting Packages
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Machine Learning and Packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy
import pandas as pd 
import numpy as np

# System Packages
import pickle









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
    
    









































