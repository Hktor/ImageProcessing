""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from numpy          import linalg as LA
from scipy.signal   import convolve2d, convolve
from scipy.ndimage  import rotate
from scipy.misc     import imresize
from skimage        import color, io
from hw2_ex1_Hector_Alvarez import gauss1d
from hw2_ex1_Hector_Alvarez import gauss2d
from hw2_ex1_Hector_Alvarez import myconv2
import pdb

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpeg')
img = color.rgb2gray(img)

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k):
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    ### your code should go here ###
    dx = np.array([-1,0,1]).reshape(1,3)
    dy = np.array([[-1],[0],[1]]).reshape(3,1)

    gx_1dfilt = gauss1d(sigma,w_size)
    gy_1dfilt = gauss1d(sigma,w_size)
    gdx = myconv2(dx,gx_1dfilt)
    gdy = myconv2(gy_1dfilt,dy)

    print ('Gdx: ' + str(gdx))
    print ('Gdy: ' + str(gdy))

    tx = np.array(image,copy=True)
    ty = np.array(image,copy=True)

    ix = convolve2d(tx,dx, mode = 'same')
    iy = convolve2d(ty,dy, mode = 'same')    

    img_row = np.size(image,0)
    img_clm = np.size(image,1)
    
    if (img_row>img_clm):

        ixx = np.zeros((img_row,img_row))
        ixx[:img_row,:img_clm] = ix[::]
        iyy = np.zeros((img_row,img_row))
        iyy[:img_row,:img_clm] = iy[::]

        ix2 = np.sum(ixx*ixx)
        iy2 = np.sum(iyy*iyy)
        ixy = np.sum(ixx*iyy)

    elif (img_row<img_clm):

        ixx = np.zeros((img_clm,img_clm))
        ixx[:img_row,:img_clm] = ix[::]
        iyy = np.zeros((img_clm,img_clm))
        iyy[:img_row,:img_clm] = iy[::]

        ix2 = ixx*ixx
        iy2 = iyy*iyy
        ixy = ixx*iyy

    else:
        ix2 = ix*ix
        iy2 = iy*iy
        ixy = ix*iy
    
    R = np.zeros ((image.shape))
    if (w_size%2 != 0):
        center = int(w_size*.5)+1
    else:
        center = int(w_size*.5)
    print (image.shape)

    row_size = np.size(image,0)
    clm_size = np.size(image,1)

    for y in range (center,row_size-center):
        for x in range (center,clm_size-center):

            sxx = np.sum(ix2[y-center:y+center+1, x-center:x+center+1])
            sxy = np.sum(ixy[y-center:y+center+1, x-center:x+center+1])
            syy = np.sum(iy2[y-center:y+center+1, x-center:x+center+1])

            det = sxx*syy - sxy**2
            trace = sxx + syy

            # M_harris = np.array(([sxx,sxy],[sxy,syy]))
            # w, v = LA.eig(M_harris)
            # det = w.item(0)*w.item(1)
            # trace = w.item(0)+w.item(1)

            R[y,x] = det - k*trace**2
            
    print (np.amax(R))
    print (np.amin(R))
    print (np.mean(R))

    return R

# 3.2
# Evaluate myharris on the image
R = myharris(img, 10, 2, 0.1)
plt.imshow(R)
plt.colorbar()
plt.show()


# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
copy_image = np.array(img, copy=True)
R_rotated =  myharris(rotate(copy_image, angle=45), 10, 2, 0.1) 
plt.imshow(R_rotated)
plt.colorbar()
plt.show()

# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
#R_scaled =      ### your code should go here ###
copy_image = np.array(img, copy=True)
R_scaled =  myharris(imresize(copy_image, 0.5), 10, 2, 0.1)
plt.imshow(R_scaled)
plt.colorbar()
plt.show()

# 3.5
# [6 Points] Looking at the results from (3.2), (3.3) and (3.4) what can we say about the
# properties of Harris corners? What is maintained? What is it invariant to? Why is that the
# case?

# Discussion:

# In the first plot, R has a range of from (-59.41, 113.91), where the edges are shown with
# the lowest negative values (dark color), and the corners with the highest positive values 
# (white color). For the second plot, while rotating the image, an increase in size could 
# be observed. The original image size is (833, 1280), and it is transformed to (1494, 1494). 
# Using the same settings for the filter size, sigma, and the factor k, R has a range on values
# (-74.91, 184.28), which are similar to the previous result. Finally, the third plot scales
# the image by a factor of 2 before using the Harris Corner detection algorithm resulting in
# a range of values of R (-6.97e11, 2.75e11). The last result is completely different to any
# of the previous plots. From the results obtained we could state the following:

# 1. Harris corner detection is invariant to rotation, but not to scaling.
# 2. Rotation is leading to an improved corner detection, however the computation required
#    could increase significantly.
# 3. The factor k has a direct influence on the result, as it can be seen in the formula
#    R[y,x] = det - k*trace**2. High values for k lead to a negative response.
# 4. The bigger size and the higher the variation for the gaussian filter (blur effect), 
#    the thicker the edges, and corners.
# 5. Downsizing the image requires less computation effort.
# 6. The window size, indicates the amount of pixels which will be weighted while building
#    the matrix H. Thus, the window size has an impact on the computation time, additionally,
#    choosing a wide window would increase the size of the edges, corners, and the peak 
#    values.


# Conclusion:

# Harris corner detection is a fast an efficient method to find corners using gradients.
# If the image is downsized and the window size remains the same, the window will cover 
# more pixels than before, thus, the range of values will increase significantly. 
# It is important to set the initial values properly to have accurate results, otherwise,
# the detection could lead to wrong analysis. 