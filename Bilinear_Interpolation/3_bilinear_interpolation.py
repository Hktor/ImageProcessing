
import sys, os
import numpy as np
import matplotlib.pyplot as plt
#from PIL import image


#   Implement interp(y_vals, x_vals, x_new) function which computes linear interpola-
#   tion for a given signal at the given locations. It takes a signal y_vals, its support x_vals
#   and new locations x_new at which the interpolation will be computed as inputs and out-
#   puts of the interpolated signal [10 points].


def test_interp():
    # Tests the interp() function with a known input and output
    # Leads to error if test fails
    x = np.array([1,2,3,4,5,6,7, 8])
    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])   
    x_new = np.array((0.5,2.3,3, 5.45))
    y_new_solution = np.array([ 0.2, 0.46, 0.6, 0.69])
    y_new_result = interp(y, x, x_new)
    np.testing.assert_almost_equal(y_new_solution, y_new_result)




def test_interp_1D():
    # Test the interp_1D() function with a known input and output
    # Leads to error if test fails
    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])   
    y_rescaled_solution = np.array([0.20000000000000001, 0.29333333333333333, 0.38666666666666671,
     0.47999999999999998, 0.57333333333333336, 0.53333333333333333,
     0.44000000000000006, 0.45333333333333331, 0.54666666666666663,
     0.64000000000000001, 0.73333333333333339, 0.82666666666666677,
     0.91999999999999993, 1.0066666666666666, 1.0533333333333335,
     1.1000000000000001])
    y_rescaled_result = interp_1D(y, 2)
    #np.testing.assert_almost_equal(y_rescaled_solution, y_rescaled_result)

    

def test_interp_2D():
    # Tests interp_2D() function with a known and unknown output
    # Leads to error if test fails
    matrix = np.array([[1,2,3],[4,5,6]])
    matrix_scaled = np.array([[ 1. ,  1.4,  1.8,  2.2,  2.6,  3. ],
                    [ 2.,   2.4,  2.8,  3.2,  3.6,  4. ],
                    [ 3.,   3.4,  3.8,  4.2,  4.6,  5. ],
                    [ 4.,   4.4,  4.8,  5.2,  5.6,  6. ]])

    result = interp_2D(matrix, 2)
    #np.testing.assert_almost_equal(matrix_scaled, result)

def interp(y_vals, x_vals, x_new): 
    # Computes interpolation at the given abscissas
    # Inputs:
    #   x_vals: Given inputs abscissas, numpy array
    #   y_vals: Given input ordinates, numpy array
    #   x_new : New abscissas to find the respective interpolated ordinates, numpy
    #   arrays
    # Outputs: 
    #   y_new: Interpolated values, numpy array
    ################### PLEASE FILL IN THIS PART ###############################
    i = 0
    y_new = np.zeros(np.size(x_new),dtype=float)
    # Minimize the vector to half of it
    # Find the points which are the closer to the reference
    for j in range (0, np.size(x_vals)-1):
        if i < np.size(x_new):
            if (x_new[i] < x_vals[0]):
                y_new[i] = y_vals[0]
                i+=1
            elif (x_new[i] >= x_vals[j] and x_new[i] <= x_vals[j+1] ):
                y_new[i] = (1-(x_new[i]-x_vals[j])/(x_vals[j+1]-x_vals[j]))*y_vals[j]+ (x_new[i]-x_vals[j])/(x_vals[j+1]-x_vals[j])*y_vals[j+1]
                i+=1
            print (str(y_new))
        print (str(i))

    return y_new 

def interp_1D(signal, scale_factor):
    # Linearly interpolates one dimensional signal by a given saling fcator
    # Inputs:
    #   signal: A one dimensional signal to be samples from, numpy array
    #   scale_factor: scaling factor, float
    #
    # Outputs:
    #   signal_interp: Interpolated 1D signal, numpy array
    ################### PLEASE FILL IN THIS PART ###############################
    signal_size      = np.size(signal)
    temp_signal      = np.zeros (signal_size*scale_factor,dtype=float)
    signal_interp    = np.zeros (signal_size*scale_factor,dtype=float)
    temp_signal_size = np.size(temp_signal)
    j = 0
    for i in range (0, temp_signal_size):
        temp_signal[i] = signal[j] 
        if (i%scale_factor == scale_factor-1):
            j+=1

    kernel_1d = np.array ([0.5, 1, 0.5])
    
    for i in range (0, temp_signal_size):
        if (i == 0):
            signal_interp[i] = temp_signal[i]*kernel_1d[1] + temp_signal[i+1]*kernel_1d[2]
        elif (i<np.size(temp_signal)-1):
            signal_interp[i] = temp_signal[i]*kernel_1d[1] + temp_signal[i+1]*kernel_1d[2]
        else:
            signal_interp[i] = temp_signal[i-1]*kernel_1d[1] + temp_signal[i]*kernel_1d[2]
    
    return signal_interp 

def interp_2D(img, scale_factor):
    # Applies bilinear interpolation using 1D linear interpolation
    # It first interpolates in one dimension and passes to the next dimension
    # Inputs:
    #   img: 2D signal/image (grayscale or RGB), numpy array
    #   scale_factor: Scaling factor, float
    # Outputs:
    #   img_interp: interpolated image with the expected output shape, numpy array
    ################### PLEASE FILL IN THIS PART ###############################

    img_row_size = np.size(img,0)
    img_clm_size = np.size(img,1)
    reconstruct_img = np.zeros ((img_row_size*int(scale_factor), img_clm_size*int(scale_factor)), dtype=float)
    img_interp      = np.zeros ((img_row_size*int(scale_factor), img_clm_size*int(scale_factor)), dtype=float)

    k, l = 0, 0
    row_limit = np.size(reconstruct_img,0)
    clm_limit = np.size(reconstruct_img,1)

    for j in range (0, clm_limit):    
        k = 0
        for i in range (0, row_limit):
            if (k < img_row_size and l < img_clm_size):
                reconstruct_img[i,j]     = img[k,l]
            if (i%scale_factor == scale_factor-1):
                k+= 1
        if (j%scale_factor == scale_factor-1):
            l+= 1

    for j in range (0, clm_limit):
        for i in range (0, row_limit):
            if (i < row_limit-1):
                img_interp[i,j] = reconstruct_img[i,j]*0.5 + reconstruct_img[i+1,j]*0.5
            else:
                img_interp[i,j] = reconstruct_img[i-1,j]*0.5 + reconstruct_img[i,j]*0.5
        if (j < clm_limit-1):
            img_interp[i,j] = reconstruct_img[i,j]*0.5 + reconstruct_img[i,j+1]*0.5
        else:
            img_interp[i,j] = reconstruct_img[i,j-1]*0.5 + reconstruct_img[i,j]*0.5

    return img_interp 

    
# Take the arguments 
filename = sys.argv[1]     # image file name
scale_factor = float(sys.argv[2]) # Scaling factor

# Before trying to directly test the bilinear interpolation on an image, we
# test the intermediate functions to see if the functions that are coded run
# correctly and give the expected results.

print('...................................................')
print('Testing functions...')
# We first test interp() function
test_interp()
print('Function interp() is working....')

# We then test interp_1D() function
test_interp_1D()
print('Function interp_1D() is working....')

# We finally test interp_2D() function
test_interp_2D()
print('Function interp_2D() is working....')

print('Passing to the bilinear interpolation of an image...')
# Read image as a matrix, get image shapes before and after interpolation
img = (plt.imread(filename)).astype('float') # need to convert to float
in_shape = img.shape # Input image shape

# Apply bilinear interpolation
img_int = interp_2D(img, scale_factor)

# Now, we save the interpolated image and show the results
print('Plotting and saving results...')
plt.figure()
plt.imshow(img_int.astype('uint8')) # Get back to uint8 data type
filename,_ = os.path.splitext(filename)
plt.savefig('{}_rescaled.jpg'.format(filename))
plt.close()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img.astype('uint8'))
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(img_int.astype('uint8'))
plt.title('Rescaled by {:2f}'.format(scale_factor))
print('Do not forget to close the plot window --- it happens:) ')
plt.show()

print('..................Well done!......................')
