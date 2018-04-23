# Plotting Libraries
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

# ML and data libraries
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import scipy
import pandas as pd 

# System libraries
import pickle 


# X = (num_samples, num_features)
def get_explained_var(X):
	# calculating Eigenvectors
	# Standardize the data
	from sklearn.preprocessing import StandardScaler
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

	return (var_exp, cum_var_exp)


def plot_explained_var(var_exp, cum_var_exp, num_features, feature_zoom):
	abs_cum_var_exp = np.abs(cum_var_exp)
	abs_var_exp = np.abs(var_exp)

	trace1 = go.Scatter(
	    x=list(range(num_features)),
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
	    x=list(range(num_features)),
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
	fig.layout.xaxis = dict(range=[0, feature_zoom], title = 'Feature columns')
	fig.layout.yaxis = dict(range=[0, 100], title = 'Explained Variance')
	fig['data'] += [go.Scatter(x= list(range(num_features)) , y=abs_cum_var_exp, xaxis='x2', yaxis='y2', name = 'Cumulative Explained Variance')]
	fig['data'] += [go.Scatter(x=list(range(num_features)), y=abs_var_exp, xaxis='x2', yaxis='y2',name = 'Individual Explained Variance')]
	py.iplot(fig, filename='inset example')



def plot_eigens(eigen_values, n_eigens, im_size):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(im_size,im_size))
    for i in list(range(n_eigens)):
    #     for offset in [10, 30,0]:
    #     plt.subplot(n_row, n_col, i + 1)
        offset =0
        plt.subplot(n_eigens, 1, i + 1)
        plt.imshow(eigen_values[i].reshape(im_size,im_size), cmap='jet')
        title_text = 'Eigenvalue ' + str(i + 1)
        plt.title(title_text, size=16)
        plt.xticks(())
        plt.yticks(())
    plt.show()



