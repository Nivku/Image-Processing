import math
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
import skimage.color
import matplotlib.pyplot as plt
import time

GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
GRAYSCALE_MAX = 255


def DFT(signal):
    """
    This function operates  discrete transform fourier on 1-D Array signal
    """
    N = signal.shape[0]
    row_base = np.arange(N)
    base = np.exp((-2 * np.pi * 1j) / N)
    base_matrix = base ** row_base
    return (np.vander(base_matrix, increasing=True)) @ signal


def IDFT(fourier_signal):
    """
    This function operates inverse discrete transform fourier on 1-D Array fourier signal

    """
    N = fourier_signal.shape[0]
    row_base = np.arange(N)
    base = np.exp((2 * np.pi * 1j) / N)
    base_matrix = base ** row_base
    return 1 / N * ((np.vander(base_matrix, increasing=True)) @ fourier_signal)


def DFT2(image):
    """
    This function operates  discrete transform fourier on 2-D Array signal
    by operating DFT on ever row / col on the image Matrix
    """
    image = np.apply_along_axis(DFT, 1, image)
    image = np.apply_along_axis(DFT, 0, image)
    return image


def IDFT2(fourier_image):
    """
    This function operates inverse discrete transform fourier on 2-D Array signal
    by operating DFT on ever row / col on the fourier image Matrix
    """
    fourier_image = np.apply_along_axis(IDFT, 1, fourier_image)
    fourier_image = np.apply_along_axis(IDFT, 0, fourier_image)
    return fourier_image


def change_rate(filename, ratio):
    """
    This function changing the duration of wav file by multiply
    the current sample rate with the ratio parameter
    """
    sample_rate, data = wav.read(filename)
    wav.write('change_rate.wav', int(sample_rate * ratio), data)


def change_samples(filename, ratio):
    """
    This function changes the duration of WAV file by reducing the
    samples using fourier
    """
    sample_rate, data = wav.read(filename)
    new_data = resize(data, ratio)
    wav.write('change_samples.wav', sample_rate, np.real(new_data))


def resize(data, ratio):
    """
    this function works on 1-D array and according to ratio
    padding with Zeros or slicing frequencies.
    """
    if ratio == 1:
        return data
    elif ratio > 1:
        slicing = math.ceil(len(data) - (len(data) / ratio))
        left = int(math.floor(slicing / 2))
        right = int(math.ceil(slicing / 2))
        fourier = DFT(data)
        fourier = np.fft.fftshift(fourier)
        fourier = fourier[left:len(fourier) - right]
        return IDFT(np.fft.ifftshift(fourier))

    else:
        padding = math.floor((len(data) / ratio) - len(data))
        right = math.ceil(padding / 2)
        left = math.floor(padding / 2)
        fourier = DFT(data)
        right = np.zeros(right)
        left = np.zeros(left)
        fourier = np.concatenate((right, fourier, left))
        return IDFT(fourier)


def resize_spectrogram(data, ratio):
    """
    this function change the duration of WAV file without changing the pitch
    by apply the resize function on every row in the spectrogram and shorten
    the rows.
    """
    spectrogram = stft(data)
    new_data = np.apply_along_axis(resize, 1, spectrogram, ratio)
    return np.real(istft(new_data))


def resize_vocoder(data, ratio):
    """
    this function change the duration of WAV file. the function using the
    phase_vocoder function to scale the spectrogram and then operates the
    resize function on every row

    """
    spectrogram = stft(data)
    new_data = phase_vocoder(spectrogram, ratio)
    return np.real(istft(new_data))


def conv_der(im):
    """
    This function calculates the derivative of x-axis, y-axis
    by using convolution. the function returns the magnitude
    """
    x_conv = np.array([0.5, 0, -0.5]).reshape(3, 1)
    dx = signal.convolve2d(im, x_conv, mode='same')
    y_conv = x_conv.reshape(1, 3)
    dy = signal.convolve2d(im, y_conv, mode='same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    """
    This function calculates the derivative of x-axis, y-axis
    by using the fourier formula. the function returns the magnitude
    """
    fourier = DFT2(im)
    fourier = np.fft.fftshift(fourier)
    N = im.shape[0]
    M = im.shape[1]
    mid_col, mid_row = np.meshgrid(np.arange(-M / 2, M / 2), np.arange(-N / 2, N / 2))
    mid_col = mid_col.astype(np.complex128)
    mid_row = mid_row.astype(np.complex128)
    mid_row *= (2 * np.pi * 1j) / N
    mid_col *= (2 * np.pi * 1j) / M
    dx = IDFT2(np.fft.ifftshift(mid_row * fourier))
    dy = IDFT2(np.fft.ifftshift(mid_col * fourier))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)

    return magnitude


###  ex2_helper  ####


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


## ex1 functions ##

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

if __name__ == '__main__':

    print(np.arange(-7//2,7//2)) ## floor
    print(np.arange(-7 /2, 7 / 2)) ##floor