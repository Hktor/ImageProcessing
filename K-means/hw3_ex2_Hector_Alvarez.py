import sys, os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

def test_kmeans():
    # Test function to see if the implemented function works correctly.
 
    num_samples = 1000
    mean = [0, 0]
    cov = [[2, 0], [0, 2]]  # diagonal covariance
    x1, y1 = np.random.multivariate_normal(mean, cov, num_samples).T
    mean = [5, 5]
    cov = [[2, 0], [0, 2]]  # diagonal covariance
    x2, y2 = np.random.multivariate_normal(mean, cov, num_samples).T

    mean = [10, 10]
    cov = [[2, 0], [0, 2]]  # diagonal covariance
    x3, y3 = np.random.multivariate_normal(mean, cov, num_samples).T

    x = np.hstack((x1,x2,x3))
    y = np.hstack((y1,y2,y3))


    img = np.vstack((x,y)).transpose()
    img = np.expand_dims(img, axis=1)
    img_clustered = cluster_with_kmeans(img, 3, max_iter = 50)
    img_clustered = img_clustered.flatten()
    
    est_means = []
    for k in range(3):
        est_means.append(np.mean(x[img_clustered == k]))
        group = np.argmin(abs(est_means[-1] - np.array([0, 5, 10])))
        
        img_clustered[group * num_samples : (group+1) * num_samples]

    maj_class = stats.mode(img_clustered[0:num_samples])[0]
    err_cl1 = np.sum(img_clustered[0:num_samples] != maj_class)
    maj_class = stats.mode(img_clustered[num_samples:2*num_samples])[0]
    err_cl2 = np.sum(img_clustered[num_samples:2*num_samples] != maj_class)
    maj_class = stats.mode(img_clustered[2*num_samples:])[0]
    err_cl3 = np.sum(img_clustered[2*num_samples:] != maj_class)
    
    err = (err_cl1 + err_cl2 + err_cl3)/(3*num_samples) * 100
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(x1,y1, color = 'red')
    plt.scatter(x2,y2, color = 'blue')
    plt.scatter(x3,y3, color = 'green')
    plt.title('Original data')
    plt.subplot(1,2,2)
    plt.scatter(x[img_clustered == 0],y[img_clustered == 0], color = 'red')
    plt.scatter(x[img_clustered == 1],y[img_clustered == 1], color = 'blue')
    plt.scatter(x[img_clustered == 2],y[img_clustered == 2], color = 'green')
    plt.title('Clustered data')
    plt.show()

    if err > 2:
        raise Exception('There may be an error in your k-means implementation.')

def assign_clusters(img_flat, centroids): 
    # Fuction implementing clustering assigmnet step of k-means algorithm. 
    # It returns the assigned clusters (clusters) and the distances of each pixel to each
    # of the clusters (dist_matrix).
    #
    # Inputs:
    #   img_flat  : Flattened image, numpy array, size of (ncols * nrows, nchannels)
    #   centroids : Centorids (means) of the clusters,numpy array, size of
    #               (number of clusters, nchannels)
    # Outputs:
    #   clusters  : Assigned clusters from 0 to K-1 (K: number of clusters),
    #               size of (ncols * nrows, 1)
    #   dist_matrix : Distance matrix, numpy array, size of (nrows * ncols,
    #                 number of clusters) 

    c_num = np.size(centroids, 0)
    img_size, cnn = img_flat.shape
    clusters = np.zeros(img_size)
    dist_matrix = np.zeros((img_size, c_num))

    k = 0
    for c in centroids:
        centers = np.repeat(c, img_size).reshape(img_size, cnn)
        dist_matrix[:, k] = np.sum((img_flat - centers)**2, axis = 1)
        k += 1
    dist_matrix = np.sqrt(dist_matrix)
    clusters = np.argmin(dist_matrix, axis=1)

    return clusters, dist_matrix

def update_centroids(img_flat, clusters):
    # Function that implements update step of k-means algorithm.
    # It returns the centroids (means) that are updated.
    # 
    # Inputs: 
    #   img_flat  : Flattened image, numpy array, size of (ncols * nrows, nchannels)
    #   clusters  : Clusters to which pixels were assigned, size of (ncols *
    #               nrows, 1)
    #
    # Outputs:
    #   centroids : Centorids (means) of the clusters,numpy array, size of
    #               (number of clusters, nchannels)
    
    c_num = np.amax(clusters,0) + 1
    chann = np.size(img_flat, 1)
    centroids = np.zeros((c_num,chann))

    for d in range(0,chann):
        for c in range(0,c_num):
            mask = np.equal(clusters,c)
            centroids[c,d] = np.mean(img_flat[mask,d])
    
    return centroids 


def cluster_with_kmeans(img, K, max_iter = 1e4, tol = 1e-4):
    # This is the function that takes an input image with one or more channels
    # and runs k-means algorithm on it to cluster its pixels
    # It iteratively calls assign_clusters and update_centroids functions 
    # and returns the clsutered image.
    #
    # Inputs:
    #   img     : Image of size (nrows, ncols) or (nrows, ncols, nchannels)
    #             Numpy array
    #   K       : Number of clusters, integer
    #   max_iter: Maximum number of iterations given to k-means algorithm, integer
   
   
    # Seed to random state in order to get the same results
    random.seed(70)
    
    # Image size to be checked -- is it a multi-channel image?
    if len(img.shape) < 3: # One-channel
        nrow, ncol = img.shape
        nchannel = 1
        img_flat = np.reshape(img, (img.size,1))
        
    else: # Multi-channel
        nrow, ncol, nchannel = img.shape
        img_flat = np.reshape(img, (nrow*ncol, nchannel))
    
    # Initialize the centroids, i.e. means
    init_centroids = img_flat[random.sample(range(nrow*ncol), K),:]
    centroids = init_centroids
    dist_prev = 0

    # Iterate between assignement and updates steps
    for iter in range(max_iter):
        
        # Assignment step
        clusters, dist_matrix = assign_clusters(img_flat, centroids)

        # Update step
        centroids = update_centroids(img_flat, clusters)

        # Do assignments not change much? If so, stop.
        if(abs(np.sum(dist_matrix) - dist_prev) < tol):
            break

        # This is to compare the new cluster assignments with the previous assignment
        dist_prev = np.sum(dist_matrix)
        
    # Reshape clusters back into the image shape
    img_clustered = np.reshape(clusters, (nrow, ncol))

    print('Converged in {:d} iterations...'.format(iter))
    # Return clustered image
    return img_clustered


def main(argv):
    # The image file and the number of intented clusters will be given from keyboard
    filename = argv[1] # Filename
    K = int(argv[2])   # Number of clusters

    # Get the file name and its extension
    file_name,file_ext = os.path.splitext(filename)

    # First test your function works correctly!
    print('\n***********************************************************')
    print('Testing your k-means implementation...')
    test_kmeans()
    print('Good job! Your k-means implementation is working fine...')
    print('************************************************************')

    # Read the input file
    if file_ext == '.npz': # If given as a .npz file (e.g. filter bank responses from the previous question)
        img_idx = 0 # Select one image from texture images, i.e. change 0 to 5 as you wish
        img = np.load(filename)['feats'][0]
    else: # if image, convert it to float
        img = (plt.imread(filename)).astype('float')

    # Cluster with kmeans algorithm
    print('Passing to clustering a given image...')
    print('Running k-means algorithm...')
    img_clustered = cluster_with_kmeans(img, K, max_iter = 50)

    # Save the results
    print('Saving results...')
    plt.figure()
    plt.imshow(img_clustered.astype('uint8'))
    plt.savefig('{}_clustered.jpg'.format(file_name))
    plt.close()

    # Show the results
    print('Plotting results...')
        
    plt.figure()
    plt.subplot(1,2,1)
    if file_ext == '.npz': # Read the original image to which the filter bank was applied.
        img = plt.imread('./data-textures/0{}.png'.format(img_idx))
        plt.imshow(img)
    else:
        plt.imshow(img.astype('uint8'))
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(img_clustered.astype('uint8'))
    plt.title('Clustered')
    print('Close the window, please!')
    plt.show()
    print('Thanks:)')

if __name__ == "__main__":
    main(sys.argv)




