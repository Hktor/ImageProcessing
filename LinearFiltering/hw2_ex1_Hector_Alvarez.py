""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams['image.cmap'] = 'gray'
import time
import pdb

img = plt.imread('cat.jpg').astype(np.float32)

plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.show()

# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn
    ### your code should go here ###
    box_filter = np.ones ((n,n))/(n*n)
    return box_filter

# 1.2
# Implement full convolution
def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    img_row = int (np.size(image,0))
    img_clm = int (np.size(image,1))
    conv = np.array(image,copy=True)

    inv_filt    = np.array(np.fliplr(filt[::-1]),copy=True)
    invfilt_row = np.size(inv_filt,0)
    invfilt_clm = np.size(inv_filt,1)

    if (img_clm>1 and img_row>1):
        if (invfilt_row%2 == 0):
            padding_y = int(invfilt_row*.5)
        else:
            padding_y = int(invfilt_row*.5)+1
        if (invfilt_clm%2 == 0 ):
            padding_x = int(invfilt_clm*.5)
        else:
            padding_x = int(invfilt_clm*.5)+1

        padding_x1 = padding_x-(padding_x-1)
        padding_x2 = padding_x+(padding_x-1)

        padding_y1 = padding_y-(padding_y-1)
        padding_y2 = padding_y+(padding_y-1)

        zero_padding = np.zeros((img_row+padding_y*2, img_clm+padding_x*2))
        pad_row = np.size(zero_padding,0)
        pad_clm = np.size(zero_padding,1)
        
        i, j = 0, 0
        for row in range (padding_y,pad_row-padding_y):
            for clm in range (padding_x,pad_clm-padding_x):
                zero_padding[row,clm]= conv[i,j]
                if (j < img_clm-1):
                    j+=1
                elif (i < img_row-1):
                    j=0
                    i+=1
                else:
                    break
          
        pad_list = []

        for row in range (padding_y1, pad_row-padding_y2):
            for clm in range (padding_x1, pad_clm-padding_x2):
                pad_list.append(zero_padding[row:row+invfilt_row,clm:clm+invfilt_clm])

        prefilt_img = []
        i=0
        for patch in pad_list:
            result = np.sum(np.multiply(patch,inv_filt))
            prefilt_img.append(result)
        
        filtered_img = np.array(prefilt_img).reshape(img_row,img_clm)
    else:
        if (img_clm == 1 and invfilt_clm == 1):
            inv_filt = np.transpose(inv_filt)
        elif (img_row == 1 and invfilt_row == 1):
            inv_filt = np.transpose(inv_filt)
        filtered_img = np.multiply(image,inv_filt)

    return filtered_img


# 1.3
# create a boxfilter of size 10 and convolve this filter with your image - show the result
bsize = 10
knel = boxfilter(bsize)
image_con = myconv2(img,knel)
print('Plotting convolution')
plt.imshow(image_con)
plt.title('convolution image')
plt.show()
### your code should go here ###

# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###

    if (filter_length%2==0):
        filter_length+=1
    #border = filter_length - filter_length*.5
    x = np.linspace(-filter_length*.5,filter_length*.5,num = filter_length).reshape(1,filter_length)
    print (x.shape)
    gaussian = np.exp(-(x*x)/(2*sigma*sigma))
    gauss_filter = gaussian/np.sum(gaussian)

    #print (gauss_filter)
    return gauss_filter


# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    ### your code should go here ###
    gaussian = gauss1d(sigma, filter_size)
    gauss2d_filter = np.zeros((filter_size,filter_size))
    gauss2d_filter = myconv2(gaussian, np.transpose(gaussian))
    #print (gauss2d_filter)
    return gauss2d_filter

# Display a plot using sigma = 3
sigma = 3

### your code should go here ###
gauss_filt = gauss2d(sigma,)

print('Plotting gaussing filter')
fig  = plt.figure()
ax   = fig.gca(projection='3d')
x = np.linspace(0,np.size(gauss_filt,0)-1, np.size(gauss_filt,0))
x, y = np.meshgrid(x,x)
surf = ax.plot_surface(x, y, gauss_filt, cmap=cm.coolwarm, linewidth=1, antialiased=False)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian filter')
fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
plt.show()


# 1.6
# Convoltion with gaussian filter
def gconv(image, sigma):
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    ### your code should go here ###
    #Filte size = 20
    gauss_filt = gauss2d(sigma,20)
    img_filtered = myconv2(image,gauss_filt)
    #print (img_filtered)
    return img_filtered


# run your gconv on the image for sigma=3 and display the result
sigma = 3

### your code should go here ###

image_gauss = gconv(img,sigma)
plt.imshow(image_gauss)
plt.title('gaussian image')
plt.show()

# 1.7
# Convolution with a 2D Gaussian filter is not the most efficient way
# to perform Gaussian convolution with an image. In a few sentences, explain how
# this could be implemented more efficiently and why this would be faster.
#
# HINT: How can we use 1D Gaussians?

### your explanation should go here ###

# A multiplication in 2 Dimensions requires nxn operations two 
# complete the task.
# A much efficient way to implement such a task is to multiply
# a vector in one dimension (suppose x), followed by a second 
# multiplication by a vector in a different dimension  (suppose y)
# Performing this method we decrease the number of operations:
#
#       | 1   3   5  |               | 1 |                 
#   M = | 2   6   10 |      = u * v =| 2 | [1     3     5]
#       | 4   12  20 |               | 4 |                  
#       
# For this example, we decrease the number of operations from 9 to 6;
# thus, the computation time is decreased achieving the same result
# faster.

# 1.8
# Computation time vs filter size experiment
size_range = np.arange(3, 100, 5)
t1d = []
t2d = []
n = 10
box = boxfilter(n)
oned_u = np.ones(n)/n
oned_v = np.transpose(oned_u)

    ### your code should go here ###

# for size in size_range:
#     box = boxfilter(size)
#     tim1 = time.clock()
#     myconv2(img,box)
#     tim2 = time.clock()
#     proctim =tim2-tim1
#     t2d.append(proctim)

#     oned_u = np.ones(size).reshape(1,size)/size
#     oned_v = np.transpose(oned_u)
#     tim3 = time.clock()
#     myconv2(img,oned_u)
#     myconv2(img,oned_v)
#     tim4 = time.clock()
#     proctim =tim4-tim3
#     t1d.append(proctim)

# # plot the comparison of the time needed for each of the two convolution cases

# plt.plot(size_range, t1d, label='1D filtering')
# plt.plot(size_range, t2d, label='2D filtering')
# plt.xlabel('Filter size')
# plt.ylabel('Computation time')
# plt.legend(loc=0)
# plt.show()
