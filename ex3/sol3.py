import numpy as np
from scipy import ndimage
import scipy as sc
from imageio import imread
import skimage.color
import os
import matplotlib.pyplot as plt

GRAYSCALE_MAX = 255
GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
MIN_DM = 16


def build_filter(filter_size):
    """
     build the normalized filter_vec according to filter_size
    :param filter_size:
    :return: filter_vec
    """
    base_vec = np.array([1, 1])
    filter_vec = base_vec
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, base_vec)
    filter_vec = filter_vec / filter_vec.sum()
    return filter_vec.reshape((1, filter_size))


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    im = ndimage.convolve(im, blur_filter)  # Row conv
    im = ndimage.convolve(im, blur_filter.T)  # Cols conv
    im = im[::2]  # Reduce rows
    im = im.transpose()
    im = im[::2]
    return im.transpose()


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """

    expand_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    expand_im[::2, ::2] = im
    expand_im = ndimage.convolve(expand_im, 2 * blur_filter)
    expand_im = ndimage.convolve(expand_im, (2 * blur_filter).T)
    # TODO - Not sure about transpose operation
    return expand_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter_vec = build_filter(filter_size)
    pyr = []
    pyr.append(im)
    for i in range(max_levels - 1):
        current_im = reduce(pyr[-1], filter_vec)
        if current_im.shape[0] < MIN_DM or current_im.shape[1] < MIN_DM:  # todo - <= OR <
            break
        pyr.append(current_im)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gauss_pyr)):
        if i == len(gauss_pyr) - 1:
            pyr.append(gauss_pyr[i])
            break
        current_expand = expand(gauss_pyr[i + 1], filter_vec)
        current_laplacian = gauss_pyr[i] - current_expand
        pyr.append(current_laplacian)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    mult_array = []
    for i in range(len(coeff)):
        mult_array.append(lpyr[i] * coeff[i])
    image = np.array(mult_array[-1])
    for j in range(len(coeff) - 2, -1, -1):
        image = expand(image, filter_vec) + mult_array[j]
    return image


def stretch_image(im):
    """
    stretches the image values to
    be between [0,1]
    :param im: the image
    :return: im
    """
    max_val = im.max()
    min_val = im.min()
    im = im - min_val
    im = im / (max_val - min_val)
    return im


def zero_matrix(rows, cols):
    """
    creates zero matrix in shape rows x cols
    :param rows: number of rows
    :param cols: number of cols
    :return: zero matrix rows x cols
    """
    return np.zeros((rows, cols))


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    new_level = min(len(pyr),levels)
    pyramid = stretch_image(pyr[0])
    for i in range(new_level - 1):
        pyr[i + 1] = stretch_image(pyr[i + 1])
        zero_mat = zero_matrix((pyr[0]).shape[0] - (pyr[i + 1]).shape[0], (pyr[i + 1]).shape[1])
        mat = np.concatenate((pyr[i + 1], zero_mat))
        pyramid = np.concatenate((pyramid, mat), axis=1)
    return pyramid


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    pyramid = render_pyramid(pyr, levels)
    plt.imshow(pyramid, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    im1_laplacian, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_laplacian, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gaussian, filter_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for i in range(len(im1_laplacian)):
        l_out.append(mask_gaussian[i] * im1_laplacian[i] + (1 - mask_gaussian[i]) * im2_laplacian[i])
    im_blend = laplacian_to_image(l_out, filter_vec, [1] * len(im1_laplacian))
    im_blend[im_blend > 1] = 1
    im_blend[im_blend < 0] = 0
    return im_blend


def rgb_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
        creates rgb pyramid blending by applying it on every colour channel
    :param im1: first image
    :param im2: second image
    :param mask: the mask
    :param max_levels:
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    """
    r_channel = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    g_channel = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    b_channel = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    rgb_im = np.dstack((r_channel, g_channel, b_channel))
    return rgb_im


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def display_blending(im1,im2,mask,blend_im):
    """
    display all the images in the blending Process
    :param im1:
    :param im2:
    :param mask:
    :param blend_im:
    """
    figure = plt.figure()
    figure.add_subplot(2, 2, 1)
    plt.imshow(im1)
    figure.add_subplot(2, 2, 2)
    plt.imshow(im2)
    figure.add_subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    figure.add_subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/shmantafim.jpg'), 2)
    im2 = read_image(relpath('externals/train.jpg'), 2)
    mask = ((read_image(relpath('externals/mask.jpg'), 1)).astype(dtype=bool))
    im_blend = rgb_blending(im1, im2, mask, 4, 8, 8)
    display_blending(im1, im2, mask, im_blend)
    return im1,im2,mask,im_blend


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/cliff.jpg'), 2)
    im2 = read_image(relpath('externals/cave.jpg'), 2)
    mask = (read_image(relpath('externals/mask2.jpg'), 1)).astype(dtype=bool)

    im_blend = rgb_blending(im1, im2, mask, 3, 4, 4)
    display_blending(im1,im2,mask,im_blend)
    return im1,im2,mask,im_blend



### EX1 ###
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


