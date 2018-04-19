import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# an extended version of an image, containing also the range of the color channels, and
# functions to manipulate it.
# the class has the following attributes:
# image_box.img       : a masked array containing the original image with the valid class pixels unmasked
# image_box.ch_range  : a list of length 3, one value per color channel containing the range of values for this channel
# and the following functions:
# image_box.find_max_range()                  : finds which channel has the maximum range, and returns the range and the channel
# image_box.split(split_point, ch_to_split)   : splits the image of the object (image_box.img) into two images. The splitting
#                                               value and channel are passed as arguments. The function returns two masked
#                                               numpy arrays, as the result of the split
class image_box:

    def __init__(self, img):
        self.img = img
        self.ch_range = np.empty((3))           # range of RGB color values (default: max range)    self.range = [255, 255, 255]
        self.ch_range[0] = ma.ptp(img[..., 0])    # range of red values
        self.ch_range[1] = ma.ptp(img[..., 1])    # range of green values
        self.ch_range[2] = ma.ptp(img[..., 2])    # range of blue value

    def find_max_range(self):
        print ('Inside Find Max Range')
        max_range_ch = self.ch_range.argmax()
        max_range    = self.ch_range[max_range_ch]
        print ('Max Value: ' + str(max_range) + '   Channel: ' + str (max_range_ch))
        return max_range, max_range_ch

    def split(self, split_point, ch_to_split):
        img1 = np.ma.array(self.img, mask=False, fill_value=0)
        img2 = np.ma.array(self.img, mask=False, fill_value=0)

        ch_to_split = int(ch_to_split)
        img1[:,:,ch_to_split] = ma.masked_less(img1[:,:,ch_to_split], split_point)
        if ch_to_split == 0:
            img1[ma.getmask(img1[:,:,ch_to_split]),1] = ma.masked
            img1[ma.getmask(img1[:,:,ch_to_split]),2] = ma.masked
        elif ch_to_split == 1:
            img1[ma.getmask(img1[:,:,ch_to_split]),0] = ma.masked
            img1[ma.getmask(img1[:,:,ch_to_split]),2] = ma.masked
        else:
            img1[ma.getmask(img1[:,:,ch_to_split]),1] = ma.masked
            img1[ma.getmask(img1[:,:,ch_to_split]),0] = ma.masked


        img2[:,:,ch_to_split] = ma.masked_greater_equal(img2[:,:,ch_to_split], split_point)
        if ch_to_split == 0:
            img2[ma.getmask(img2[:,:,ch_to_split]),1] = ma.masked
            img2[ma.getmask(img2[:,:,ch_to_split]),2] = ma.masked
        elif ch_to_split == 1:
            img2[ma.getmask(img2[:,:,ch_to_split]),0] = ma.masked
            img2[ma.getmask(img2[:,:,ch_to_split]),2] = ma.masked
        else:
            img2[ma.getmask(img2[:,:,ch_to_split]),1] = ma.masked
            img2[ma.getmask(img2[:,:,ch_to_split]),0] = ma.masked

        #print ('Image 1 after : ' + str(img1))
        #print ('Image 2 after : ' + str(img2))

        return img1, img2


def find_split_point(images):
    # Finds the image with the max range. It returns the index of this image and the corresponding color channel.
    # INPUTS  -  @ images:   a list of candidates for a split. Each item should be of type: image_box
    # OUTPUTS -  a tuple (index_of_image_to_split, channel_of_maximum_range)
    maxime, n = 0, 0
    print ('Inside Find Split Point...')

    if len(images)<=1:
        for i in range (len(images)):
            max_range, max_range_ch = images[i].find_max_range()
            if (maxime  < max_range):       
                maxime  = max_range         
                ch_to_split  = max_range_ch
                img_to_split_ind = i
    else:
        for i in range (len(images)):
            max_range, max_range_ch = images[i].find_max_range()
            if (maxime  < max_range):       
                maxime  = max_range         
                ch_to_split  = max_range_ch
                img_to_split_ind = i

    #print ('Image to split: ' + str(img_to_split_ind) + '    Channel: ' + str(ch_to_split))
    return img_to_split_ind, ch_to_split


def find_split_value(img_box, ch):
    split_point = np.ma.median(img_box.img[:,:,ch])
    #print (split_point)
    return split_point

def reconstruct_quantized_image(images):
    # returns the final quantized image, as a simple numpy array (not masked) of type numpy.uint8, where each
    # pixel value is replaced by the mean value of the pixels of the class it belongs to

    q_img = np.zeros(images[0].img.shape, dtype=np.uint8)
    print (q_img.shape)

    for box in images:
        for color in range (box.img.shape[2]):
            average = ma.mean(box.img[:,:,color])
            q_img[~ma.getmask(box.img[:,:,color]),color] = average

    return q_img


def hw1_q4(orig_img, N):
    images_q = []
    orig_img = np.ma.array(orig_img, mask=False, fill_value=0)
    images_q.append(image_box(orig_img)),
    k = 0 
    while (k<N): 
        split_ind, ch_to_split     = find_split_point(images_q)    
        split_point                = find_split_value(images_q[split_ind], ch_to_split)
        print ('Find split value successful')
        new_box1, new_box2  = images_q[split_ind].split(split_point, ch_to_split)
        del images_q[split_ind]
        images_q.append(image_box(new_box1)) 
        images_q.append(image_box(new_box2)) 
        print ('Split function successful')
        k+=2
    q_img = reconstruct_quantized_image(images_q)
    return q_img

# ---------------------------------------------MAIN SCRIPT-------------------------------------------------------------------#
# initialize
# number of color classes
N = 20
# path to the image
image_path = 'monkey_face.jpg'
# load the image
orig_img = plt.imread(image_path).astype(np.uint8)

q_img = hw1_q4(orig_img, N)

# show the original image next to the color-quantized image, for visual inspection of the result
print('Showing the results...')
plt.subplot(1, 2, 1)
plt.imshow(orig_img)
plt.axis('off')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(q_img)
plt.axis('off')
plt.title('Quantized image with ' + str(N) + ' colors')
plt.show()
print('You are finished with question 4!')
