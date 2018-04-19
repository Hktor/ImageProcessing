# HW3 Question 1 
# ISIP 2017: Raphael Sznitman

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import hw3_ex1_makeLMfilter_Hector_Alvarez as lm
from scipy import signal
# from scipy.signal import convolve2d, convolve
import glob
import os

plt.rcParams['image.cmap'] = 'gray'

# (1.1) Create LM Filters and display them.
support_size = 49
F = lm.makeLMfilters(support_size)

#Check filters by plotting

flt_num = np.size(F, axis = 2)
print(flt_num)
plt_clm = int(flt_num/4)
plt_row = int(flt_num/plt_clm)

for i in range (0,flt_num):
    plt.subplot(plt_row,plt_clm,i+1)             # the first subplot in the first figure
    plt.imshow(F[:,:,i])
    plt.axis('off')
plt.show()

# (1.2) Convolve filter-bank with each image in the dataset

def calc_features(im_files, F):
    #Calculates for each image in im_files the pixel-wise response of
    # LM filter-bank F of size (S,S,48). Return a list of (M,N,48) numpy array
    feats = []
    
    count = 0
    for image in im_files:
        img = plt.imread(image).astype(np.float)

        print(image)
        M = np.size(img,0)
        N = np.size(img,1)
        i_cnv = np.zeros((M, N, flt_num))
        print(i_cnv.shape)
        for flt in range (0,flt_num):
            filt = np.array(F[:, :, flt], copy=True)
            i_cnv[:, :, flt] = signal.convolve2d(img,filt, 'same')

        feats.append(i_cnv)

    return feats

#Extract file names
files = glob.glob(os.path.join('data-textures','*.png'))
files = sorted(files)

print('Convolve on ' + str(len(files)) +  ' images...')
feats = calc_features(files,F)
np.savez('q1_resp', feats=feats) #Save data to file
print('...done')

# (1.3) Reshape and aggregate the filter responses.

feats = np.reshape(feats, (np.size(files), -1, flt_num))

def make_histograms(feats):
    #Returns feature histograms from list of feature vectors. Input feats is a list of (M,N,48) arrays.
    img_num = np.size(feats,0)
    H = np.zeros((img_num, flt_num))
    for i in range(0,img_num):
        for j in range(0,flt_num):
            H[i, j] = np.sum(np.abs(feats[i, :, j]))
        H[i,:] = H[i,:]/np.sum(np.abs(H[i,:]))

    return H

H = make_histograms(feats)

#Plot histograms
# ...
hnum = np.size(H, 0)
bin_num = np.linspace(1, flt_num, flt_num)
c_img = 0
wth = 0.75

color=iter(cm.rainbow(np.linspace(0,1,hnum)))
for img in H:
    c=next(color)
    plt.subplot(1, 6, c_img+1)
    plt.bar(bin_num, img, wth, label=files[c_img], color = c)
    plt.legend(loc='upper right')
    c_img+=1

plt.show()

c_img = 0
wth = 0.05
color=iter(cm.rainbow(np.linspace(0,1,hnum)))
for img in H:
    c=next(color)
    plt.subplot(111)
    plt.bar(bin_num, img, wth, label=files[c_img], color = c)
    plt.legend(loc='upper right')
    bin_num+=0.1
    c_img+=1

plt.show()


# (1.4) Evaluate full matrix of possible comparisons. Use L2 norm to compare.

h_size = np.size(H,0)
s_cmp = int(h_size/2)
L2distance = np.zeros((s_cmp,s_cmp))


for i in range(0, s_cmp):
    for j in range(0, s_cmp):
        #L2distance[i, j]   = np.sqrt(np.abs(np.sum(H[i,:]**2) - np.sum(H[s_cmp+j,:]**2)))
        L2distance[i, j]   = np.sqrt(np.abs((np.sum((H[i,:] - H[s_cmp+j,:])**2))))


plt.imshow(L2distance)
plt.show()

plt.imshow(L2distance, cmap='Greys')
plt.show()

# (1.5) Your answer here.
# [2 Points] Explain in a few sentences the pattern observed. Why are there low values and
# what do they correspond to?

# Answer:
# The pattern observed describes the relation between the images. The dark color
# has the lowest value, meaning that the difference is near zero, thus both images
# are identical. The other colors correspond to varations with higher values for
# the L2 distance between images indicating the pictures are not identical. This
# method is useful for having a graphical representation.

# IMPORTANT: WHEN CHANGING THE COLOR MAP CONFIGURATION OF THE PLOT, THE COLOR
# REPRESENTATION CHANGES, SUCH IS THE CASE OF USING " cmap='Greys' ".
