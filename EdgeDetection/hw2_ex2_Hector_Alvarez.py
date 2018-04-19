""" 2 Finding edges """

import numpy as np
import numpy.ma as ma
from skimage import color, io
from hw2_ex1_Hector_Alvarez import gauss1d
from hw2_ex1_Hector_Alvarez import gauss2d
from hw2_ex1_Hector_Alvarez import gconv
from hw2_ex1_Hector_Alvarez import myconv2
from scipy.signal   import convolve2d, convolve
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import pdb

# load image
img = io.imread('lenaTest2.jpg')
img = color.rgb2gray(img)


### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###


# 2.1
# Gradients
# define a derivative operator
dx = np.array([-1,0,1]).reshape(1,3)
dy = np.array([[-1],[0],[1]]).reshape(3,1)

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
### your code should go here ###

gx_1dfilt = gauss1d(sigma,np.size(dx))
gy_1dfilt = gauss1d(sigma,np.size(dx))
gdx = myconv2(dx,gx_1dfilt)
gdy = myconv2(gy_1dfilt,dy)

print ('Gdx: ' + str(gdx))
print ('Gdy: ' + str(gdy))

# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : geadient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    ### your code should go here ###

    tx = np.array(image,copy=True)
    ty = np.array(image,copy=True)

    ix = convolve2d(tx,dx, mode='same')
    iy = convolve2d(ty,dy, mode='same')    

    # ix = myconv2(tx,dx)
    # iy = myconv2(ty,dy) 

    img_row = np.size(image,0)
    img_clm = np.size(image,1)
    
    if (img_row>img_clm):

        ixx = np.ones((img_row,img_row))
        ixx[:img_row,:img_clm] = ix[::]
        iyy = np.zeros((img_row,img_row))
        iyy[:img_row,:img_clm] = iy[::]

        grad_mag = np.ma.sqrt(ixx**2+iyy**2)
        grad_dir = np.ma.arctan2(iyy,ixx) + np.pi*0.875

        grad_mag_image  = np.array(grad_mag[:img_row,:])
        grad_dir_image  = np.array(grad_dir[:img_row,:])

    elif (img_row<img_clm):

        ixx = np.ones((img_clm,img_clm))
        ixx[:img_row,:img_clm] = ix[::]
        iyy = np.zeros((img_clm,img_clm))
        iyy[:img_row,:img_clm] = iy[::]

        grad_mag = np.ma.sqrt(ixx**2+iyy**2)
        grad_dir = np.ma.arctan2(iyy,ixx) + np.pi*0.875

        grad_mag_image  = np.array(grad_mag[:img_row,:])
        grad_dir_image  = np.array(grad_dir[:img_row,:])

    else:
        grad_mag_image = np.sqrt(ix**2+iy**2)
        grad_dir_image = np.arctan2(iy,ix) + np.pi*0.875

    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')
plt.show()

# 2.3
# Edge images of particular directions
def make_edge_map(image, dx, dy):
    # INPUTS
    # @image        : a 2D image
    # @gdx          : gradient along x axis
    # @gdy          : geadient along y axis
    # OUTPUTS:
    # @ edge maps   : a 3D array of shape (image.shape, 8) containing the edge maps on 8 orientations

    ### your code should go here ###

    # For i = 0;    7, an edge image ei should have values 0 except 
    # if a pixel has an edge of magnitude greater than threshold, r= 3 
    # and has an orientation in the interval ( 2i pi/ 8 , - pi/ 8 , 
    # 2i pi / 8 + pi/8 ).In this case, the pixel should have value 255.

    
    edge_mag, edge_dir = create_edge_magn_image(image, dx, dy)
    threshold = 0.05*(np.amax(edge_mag)-np.amin(edge_mag))
    #threshold = 3
    edges = []

    for i in range (0,8):
        mag_img = np.zeros((edge_mag.shape))

        low_limit = (2*i*np.pi*0.125 - np.pi*0.125)
        upp_limit = (2*i*np.pi*0.125 + np.pi*0.125)

        np.place(mag_img, np.greater    (edge_mag, threshold),255)
        np.place(mag_img, np.less       (edge_dir,low_limit), 0)
        np.place(mag_img, np.greater    (edge_dir,upp_limit), 0)
        np.place(mag_img, np.less_equal (edge_mag, threshold),0)
        
        edges.append(mag_img)
    
    maps = np.array(edges, dtype=float)
    maps_rota = np.swapaxes(maps,0,1)
    edge_maps = np.swapaxes(maps_rota,1,2)


    return edge_maps


# verify with circle image
circle = plt.imread('circle.jpg')
print ('Make Edge Map... ')
edge_maps = make_edge_map(circle, dx, dy)
print ('Finished Edge Map... ')
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((circle, all_in_row), axis=1))
plt.title('Circle and edge orientations')
plt.show()

# now try with original image

print ('Make Edge Map... ')
edge_maps = make_edge_map(img, dx, dy)
print ('Finished Edge Map... ')
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((img, all_in_row), axis=1))
plt.title('Circle and edge orientations')
plt.show()

# 2.4
# Edge non max suppresion
def edge_non_max_suppression(img_edge_mag, edge_maps):
    # This function performs non maximum suppresion, in order to reduce the width of the edge response
    # INPUTS
    # @img_edge_mag   : 2d image, with the magnitude of gradients in every pixel
    # @edge_maps      : 3d image, with the edge maps
    # OUTPUTS
    # @non_max_sup    : 2d image with the non max suppresed pixels

    # ° P(i+1,j-1)              ° P(i,j+1)-------°(A)   ° P(i+1,j+1)
    #                           |           ux
    #                           | uy
    #                           |
    # ° P(i,j-1)                ° P(i,j)                ° P(i,j+1)
    #                           |
    #                        vy |
    #                           |
    # ° P(i-1,j-1)  (B)°--------° P(i-1,j)              ° P(i-1,j+1)
    #                       vx

    # P(i,j) given by the gradient magnitude in specific angle, detected with edge_maps
    # A,B are given by the normal, and the interpolation of the following rows, columns in each direction
    # The value at P(i,j) should be bigger than the values of the gradients in A and B to be a maximum
    # Ga = (ux/uy) G(i+1,j+1) + (uy-ux)/uy G(i,j+1)
    # Gb = (vx/vy) G(i-1,j-1) + (vy-vx)/vy G(i,j-1)

    # For each image in the range n(pi/8) get the points which are = 255
    # Obtain the normal to the gradient direction
    # Compute A, B
    # Calculate the gradient values
    # If G(i,i)>Ga and G(i,j)>Gb G(i,j)=255 
    i = 0
    non_max_sup   = np.zeros(img_edge_mag.shape)

    e_map = np.swapaxes(edge_maps,1,2)
    maps  = np.swapaxes(e_map,0,1)

    mag      = np.array(img_edge_mag,copy=True)
    size_row = np.size(mag,0)
    size_clm = np.size(mag,1)
    max_val = np.amax(mag)
    for i_map in maps:
        edge     = np.column_stack((np.where(i_map[::]==255)))
        
        for coordinates in edge:
            coord_y = coordinates[0]
            coord_x = coordinates[1]
            
            if (coord_y < size_row-1 and coord_x < size_clm-1):
                Ga, Gb, Gc, Gd = 0, 0, 0, 0
                max_sup = mag[coord_y,coord_x]
                # ux = 0.5
                # uy = 1.0
                # Ga = (ux/uy)*mag[coord_y+1,coord_x] + ((uy-ux)/uy)*mag[coord_y+1,coord_x-1]

                # if (i==2 or i== 6):
                #     Ga = (0.5)*mag[coord_y+1,coord_x] + (0.5)*mag[coord_y+1,coord_x-1]
                #     Gb = (0.5)*mag[coord_y-1,coord_x] + (0.5)*mag[coord_y-1,coord_x+1]
                # elif (i==3 or i== 7):
                #     Ga = (0.5)*mag[coord_y,coord_x-1] + (0.5)*mag[coord_y+1,coord_x-1]
                #     Gb = (0.5)*mag[coord_y,coord_x+1] + (0.5)*mag[coord_y-1,coord_x+1]
                # elif (i==0 or i== 4):
                #     Ga = (0.5)*mag[coord_y,coord_x-1] + (0.5)*mag[coord_y-1,coord_x-1]
                #     Gb = (0.5)*mag[coord_y,coord_x+1] + (0.5)*mag[coord_y+1,coord_x+1]
                # elif (i==1 or i== 5):
                #     Ga = (0.5)*mag[coord_y-1,coord_x] + (0.5)*mag[coord_y-1,coord_x-1]
                #     Gb = (0.5)*mag[coord_y+1,coord_x] + (0.5)*mag[coord_y+1,coord_x+1]
            
                if (i==0 or i== 4):
                    Ga = np.maximum(mag[coord_y,coord_x+1],mag[coord_y,coord_x-1])

                elif (i==1 or i== 5):
                    Ga = np.maximum(mag[coord_y+1,coord_x+1],mag[coord_y-1,coord_x-1])
                
                if (i==2 or i== 6):
                    Ga = np.maximum(mag[coord_y+1,coord_x],mag[coord_y-1,coord_x])
                
                elif (i==3 or i== 7):
                    Ga = np.maximum(mag[coord_y+1,coord_x-1],mag[coord_y-1,coord_x-1])

                if (max_sup>Ga):
                    non_max_sup[coord_y,coord_x] = max_sup
                else:
                    non_max_sup[coord_y,coord_x] = 0
        i+=1

    return non_max_sup


# show the result
img_non_max_sup = edge_non_max_suppression(img_edge_mag, edge_maps)
plt.imshow(np.concatenate((img, img_edge_mag, img_non_max_sup), axis=1))
plt.title('Original image, magnitude edge, and max suppresion')
plt.show()


# 2.5
# Canny edge detection (BONUS)
def canny_edge(image, sigma=2):
    # implementation of canny edge detector
    # INPUTS
    # @image      : 2d image
    # @sigma      : sigma parameter of gaussian
    # OUTPUTS
    # @canny_img  : 2d image of size same as image, with the result of the canny edge detection

    ### your code should go here ###

    gaux_1dfilt = gauss1d(sigma,np.size(dx))
    gauy_1dfilt = gauss1d(sigma,np.size(dy))
    gradx = myconv2(dx,gaux_1dfilt)                             # Gdx
    grady = myconv2(gauy_1dfilt,dy)                             # Gdy

    g_mag, g_dir = create_edge_magn_image(image, gradx, grady)  # gradient magnitude, direction
    ed_map = make_edge_map(image, gradx, grady)                 # make_edge_map
    m_sup = edge_non_max_suppression(g_mag, ed_map)             # non_maxima_suprema

    hg_tsh = 0.20*(np.amax(g_mag)-np.amin(g_mag))               # strong edges > high threshold = 20
    lw_tsh = 0.1*(np.amax(g_mag)-np.amin(g_mag))                # soft edges   < low threshold  = 10

    canny_img = np.zeros((image.shape))                         # output declaration

    e_map = np.swapaxes(ed_map,1,2)                             # swap axis (255,255,8) -> (255,8,255)
    maps  = np.swapaxes(e_map,0,1)                              # swap axis (255,8,255) -> (8,255,255)

    size_row = np.size(g_mag,0)
    size_clm = np.size(g_mag,1)
    max_val  = np.amax(g_mag)
    i = 0

    for i_map in maps:
        mask     = np.column_stack((np.where(i_map[::]>lw_tsh)))
        for coordinates in mask:
            y = coordinates[0]
            x = coordinates[1]

            if (i_map[y,x]>=hg_tsh):
                canny_img[y,x]=m_sup[y,x]
                
            elif (y < size_row-1 and x < size_clm-1):
                if (i==0 or i== 4):
                    if (i_map[y+1,x]>=hg_tsh or i_map[y-1,x]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                elif (i==1 or i== 5):
                    if (i_map[y+1,x-1]>=hg_tsh or i_map[y-1,x+1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                elif (i==2 or i== 6):
                    if (i_map[y,x-1]>=hg_tsh or i_map[y,x+1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                else:
                    if (i_map[y+1,x+1]>=hg_tsh or i_map[y-1,x-1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

        i+=1

    #np.place(canny_img, np.greater(canny_img,0),255)  

    return canny_img

canny_img = canny_edge(img)
plt.imshow(np.concatenate((img, canny_img), axis=1))
plt.title('Original image and canny edge detector')
plt.show()
