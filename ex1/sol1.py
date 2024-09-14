import math
import numpy as np
from imageio.v2 import imread
import skimage.color
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
GRAYSCALE_MAX = 255
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
def normalize(image):
    """
    This function changing the matrix values to float and between [0,1]
    """
    image = image.astype(np.float64)
    image /= GRAYSCALE_MAX
    return image


def read_image(filename, representation):
    """
     Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)  # Image as matrix
    image = normalize(image)  # Matrix as float and normalized
    if len(image.shape) == RGB_DIM and representation == GRAYSCALE:  # in case of RGB2GRAY
        return skimage.color.rgb2gray(image)
    return image


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
    else:
        plt.imshow(image)
        plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    yiq = np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)
    return yiq


def yiq2rgb(imYIQ):
    """
    Converts YIQ matrix to RGB representation
    """
    inverse_matrix = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
    rgb = np.dot(imYIQ, inverse_matrix.T)
    return rgb


def model_convert(im_model):
    """
    returns histogram of the im_model according to the original representation
    """
    yiq = np.empty_like(im_model)
    if len(im_model.shape) == RGB_DIM:
        yiq = rgb2yiq(im_model)
        y_channel = yiq[:, :, 0] * GRAYSCALE_MAX
        y_channel = y_channel.astype(np.int64)
        hist_orig = np.histogram(y_channel, bins=256, range=(0, 256))[0]
    else:
        hist_orig = np.histogram(im_model*GRAYSCALE_MAX, bins=256, range=(0, 256))[0]
    return yiq, hist_orig


def mapintensity(im_orig, normalized):
    """
    Map the new intensity of image after equalization algorithm
    """
    im_orig = im_orig.astype(np.int64)
    im_orig = normalized[im_orig]
    new_hist = np.histogram(im_orig,bins=256, range=(0, 256))[0]
    return im_orig, new_hist


def histogram_equalize(im_orig):
    """
    The equalization algorithm
    """
    # STEP 1 -  Check image model and compute the image histogram, The values between [0,255]
    yiq_model, hist_orig = model_convert(im_orig)


    # STEP 2 - Compute the cumulative histogram
    stretched_hist = np.cumsum(hist_orig)

    # STEP 3 - Normalize the cumulative histogram
    stretched_hist = stretched_hist / (stretched_hist[len(stretched_hist) - 1])

    # STEP 4 - Multiply the normalized histogram by the ( Z - 1 )
    stretched_hist *= GRAYSCALE_MAX
    nonzero = np.nonzero(stretched_hist)[0]
    first = stretched_hist[nonzero[0]]

    # STEP 4 - Stretching between [0,Z-1]
    stretched_hist = GRAYSCALE_MAX * ((stretched_hist - first) / (stretched_hist[GRAYSCALE_MAX] - first))

    # STEP 6 - Round up to the close integer
    stretched_hist = np.round(stretched_hist)

    # STEP  7 -  create new hist_orig to the picture according to normalized
    if len(im_orig.shape) == RGB_DIM:
        # RGB CASE
        y_channel = yiq_model[:,:,0]*GRAYSCALE_MAX
        im_eq_y, new_hist = mapintensity(y_channel, stretched_hist)
        norm_im_eq_y = normalize(im_eq_y)
        yiq_model[:, :, 0] = norm_im_eq_y
        im_eq = yiq2rgb(yiq_model)
        return [im_eq, hist_orig, new_hist]

    # B/W CASE
    else:
        im_eq_r, new_hist = mapintensity(im_orig*GRAYSCALE_MAX, stretched_hist)
        im_eq = normalize(im_eq_r)
        return [im_eq, hist_orig, new_hist]

####### PART 2 ########

def first_distribution(histogram, n_quant, pixels):
    hist_seg = []
    hist_seg.append(-1)
    avg_pixels = pixels / n_quant
    next_z = 1
    quant_n = 1
    while True:
        if n_quant == quant_n:
            hist_seg.append(GRAYSCALE_MAX)
            break
        partial_sum = np.cumsum(histogram[0:next_z])
        if partial_sum[len(partial_sum) - 1] >= quant_n * avg_pixels:
            hist_seg.append(next_z - 1)
            next_z += 1
            quant_n += 1
        else:
            next_z = next_z + 1
    return hist_seg

def compute_q(histogram, z_seg, n_quant):
    quant_seg = []
    z_quant = - 1
    for i in range(n_quant):
        numerator = 0
        denominator = 0
        while z_quant < math.floor(z_seg[i + 1]):
            numerator += histogram[z_quant + 1] * (z_quant + 1)
            denominator += histogram[z_quant + 1]
            z_quant += 1
        quant_seg.append((numerator / denominator))
    return quant_seg


def compute_z(quant_segment):
    z_seg = []
    for i in range(len(quant_segment) + 1):
        if i == 0:
            z_seg.append(-1)
        elif i == len(quant_segment):
            z_seg.append(255)
        else:
            z_seg.append((quant_segment[i - 1] + quant_segment[i]) / 2)
    return z_seg


def compute_error(z_seg, q_seg, histogram):

    total = 0
    seg_total = 0
    for i in range(len(q_seg)):
        for j in range(math.floor(z_seg[i]) + 1, math.floor(z_seg[i + 1]) + 1):
            seg_total += ((q_seg[i] - j) ** 2) * (histogram[j])
        total += seg_total
        seg_total = 0
    return total

def map(im_orig,z_seg,q_seg):
    lookup_table = np.zeros(GRAYSCALE_MAX +1)
    for i in range(len(z_seg) -1):
        lookup_table[math.floor(z_seg[i]) + 1:math.floor(z_seg[i+1]) + 1] = math.floor(q_seg[i])
    im_orig = lookup_table[im_orig]
    return im_orig

def quantize(im_orig, n_quant, n_iter):
    #Edge case - TODO -  Check what we need to return
    if n_quant == 0 or n_iter == 0:
        return

    # STEP 0 - RGB2YIQ if needed
    yiq_model, hist_orig = model_convert(im_orig)
    pixels_num = (yiq_model.shape[0]) * (yiq_model.shape[1])  # Num of pixel

    # STEP 1 - First distribution which every section has same number of pixels
    z_seg = first_distribution(hist_orig, n_quant, pixels_num)

    # STEP 2  - getting the minimum
    q_seg = []
    error = []
    for i in range(n_iter):
        q_seg = compute_q(hist_orig, z_seg, n_quant)
        z_seg = compute_z(q_seg)
        error.append(compute_error(z_seg, q_seg, hist_orig))
        if i > 0 and error[i] == error[i-1]: # Convergence case
            break

    print(q_seg)
    print(z_seg)

    # RGB CASE
    if len(im_orig.shape) == RGB_DIM:
        y_channel = yiq_model[:, :, 0] * GRAYSCALE_MAX
        y_channel = y_channel.astype(np.int64)
        im_eq_y = map(y_channel,z_seg,q_seg)
        im_eq_y_norm = normalize(im_eq_y)
        yiq_model[:, :, 0] = im_eq_y_norm
        im_eq = yiq2rgb(yiq_model)
        return [im_eq,error]

    #B/W CASE
    im_orig = GRAYSCALE_MAX*im_orig
    im_orig = im_orig.astype(np.int64)
    im_eq = map(im_orig,z_seg,q_seg)
    im_eq_norm = normalize(im_eq)
    return [im_eq_norm ,error]

def quantize_rgb(im_orig, n_quant): # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass
#
# if __name__ == '__main__':
#
#         x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
#         grad = np.tile(x, (256, 1))
#         grad1 = normalize(grad)
#         low_contrast = read_image('low_contrast.jpg', 1)
#
#         plt.imshow(grad1,cmap=plt.cm.gray)
#         plt.show()
#
#         im = histogram_equalize(grad1)[0]
#         plt.imshow(im,cmap=plt.cm.gray)
#         plt.show()
#
#
#
#         #todo  - Crashing when we use 30 iterations for the pic grad1s
#         q_p = quantize(im,20, 4)
#         plt.plot(q_p[1])
#         plt.show()
#
#
#         plt.imshow(q_p[0],cmap=plt.cm.gray)
#         plt.show()
#
