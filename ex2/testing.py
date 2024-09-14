import unittest
import numpy as np
import sol2 as sol2
import scipy.io.wavfile
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from imageio import imread

ERROR = -1


class MyTestCase(unittest.TestCase):
    def test_DTF(self):
        data = np.array([1, 2, 3]).astype(np.float64)
        my_result = sol2.DFT(data)
        good_result = np.fft.fft(data)
        self.assertEqual(my_result.shape, data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        data2 = np.array([[1], [2], [3]]).astype(np.float64)
        my_result2 = sol2.DFT(data2)
        self.assertEqual(my_result2.shape, data2.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result)))

    def test_DFT_large_data(self):
        data = np.arange(2000).astype(np.float64)
        my_result = sol2.DFT(data)
        good_result = np.fft.fft(data)
        self.assertEqual(my_result.shape, data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        data2 = np.arange(2000).reshape(2000, 1).astype(np.float64)
        my_result2 = sol2.DFT(data2)
        self.assertEqual(my_result2.shape, data2.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result)))

    def test_IDTF(self):
        fourier_data = np.fft.fft(np.array([1, 2, 3])).astype(np.complex128)

        my_result = sol2.IDFT(fourier_data)
        good_result = np.fft.ifft(fourier_data)
        self.assertEqual(my_result.shape, fourier_data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        my_result2 = sol2.IDFT(fourier_data.reshape(3, 1))
        self.assertEqual(my_result2.shape, (3, 1))
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result)))

    def test_IDFT_large_data(self):
        fourier_data = np.fft.fft(np.arange(2000)).astype(np.complex128)

        my_result = sol2.IDFT(fourier_data)
        good_result = np.fft.ifft(fourier_data)
        self.assertEqual(my_result.shape, fourier_data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        my_result2 = sol2.IDFT(fourier_data.reshape(2000, 1))
        self.assertEqual(my_result2.shape, (2000, 1))
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result)))

    def test_DTF2(self):
        data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        my_result = sol2.DFT2(data)
        good_result = np.fft.fft2(data)
        self.assertEqual(my_result.shape, data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        data2 = np.array([[[-1], [2], [-4]], [[8], [-16], [32]]]).astype(np.float64)
        my_result2 = sol2.DFT2(data2)
        good_result2 = np.fft.fft2(data2.reshape(2, 3))
        self.assertEqual(my_result2.shape, data2.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result2.flatten())))

    def test_IDTF2(self):
        fourier_data = np.fft.fft(np.array([[1, 2, 3], [4, 5, 6]])).astype(np.complex128)

        my_result = sol2.IDFT2(fourier_data)
        good_result = np.fft.ifft2(fourier_data)
        self.assertEqual(my_result.shape, fourier_data.shape)
        self.assertTrue(np.alltrue(np.isclose(my_result, good_result)))

        my_result2 = sol2.IDFT2(fourier_data.reshape(2, 3, 1))
        self.assertEqual(my_result2.shape, (2, 3, 1))
        self.assertTrue(np.alltrue(np.isclose(my_result2.flatten(), good_result.flatten())))

    def test_change_rate(self):
        wav_data_orig = scipy.io.wavfile.read(r"aria_4kHz.wav")

        sol2.change_rate(r"aria_4kHz.wav", 0.25)
        wav_data_025 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_025[1] + 1))
        self.assertEqual(int(wav_data_orig[0] / 4), wav_data_025[0])

        sol2.change_rate(r"aria_4kHz.wav", 0.8)
        wav_data_08 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_08[1] + 1))
        self.assertEqual(int(wav_data_orig[0] * 0.8), wav_data_08[0])

        sol2.change_rate(r"aria_4kHz.wav", 1)
        wav_data_1 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_1[1] + 1))
        self.assertEqual(int(wav_data_orig[0] * 1), wav_data_1[0])


        sol2.change_rate(r"aria_4kHz.wav", 2)
        wav_data_2 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_2[1] + 1))
        self.assertEqual(int(wav_data_orig[0] * 2), wav_data_2[0])

        sol2.change_rate(r"aria_4kHz.wav", 3)
        wav_data_3 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_3[1] + 1))
        self.assertEqual(int(wav_data_orig[0] * 3), wav_data_3[0])

        sol2.change_rate(r"aria_4kHz.wav", 4)
        wav_data_4 = scipy.io.wavfile.read(r"change_rate.wav")
        self.assertTrue(np.all(wav_data_orig[1] - wav_data_4[1] + 1))
        self.assertEqual(int(wav_data_orig[0] * 4), wav_data_4[0])

    def test_resize_spectrogram(self):
        wav_data_orig = scipy.io.wavfile.read(r"aria_4kHz.wav")

        # ratio = 0.25
        # print("starting test_resize_spectrogram with ratio:", ratio)
        # wav_data_025 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        # self.assertEqual(wav_data_orig[1].dtype, wav_data_025.dtype)
        # self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_025))
        # print("finished test with ration parameter:", ratio)

        # ratio = 0.8
        # print("starting test_resize_spectrogram with ratio:", ratio)
        # wav_data_08 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        # self.assertEqual(wav_data_orig[1].dtype, wav_data_08.dtype)
        # self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_08))
        # print("finished test with ration parameter:", ratio)

        ratio = 1
        print("starting test_resize_spectrogram with ratio:", ratio)
        wav_data_1 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_1.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_1))
        print("finished test with ration parameter:", ratio)

        # ratio = 2
        # print("starting test_resize_spectrogram with ratio:", ratio)
        # wav_data_2 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        # self.assertEqual(wav_data_orig[1].dtype, wav_data_2.dtype)
        # self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_2))
        # print("finished test with ration parameter:", ratio)

        ratio = 3
        print("starting test_resize_spectrogram with ratio:", ratio)
        wav_data_3 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_3.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_3))
        print("finished test with ration parameter:", ratio)

        ratio = 4
        print("starting test_resize_spectrogram with ratio:", ratio)
        wav_data_4 = sol2.resize_spectrogram(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_4.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_4))
        print("finished test with ration parameter:", ratio)

    def test_resize_vocoder(self):
        wav_data_orig = scipy.io.wavfile.read(r"aria_4kHz.wav")

        ratio = 0.25
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_025 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_025.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_025))
        print("finished test with ration parameter:", ratio)

        ratio = 0.8
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_08 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_08.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_08))
        print("finished test with ration parameter:", ratio)

        ratio = 1
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_1 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_1.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_1))
        print("finished test with ration parameter:", ratio)

        ratio = 2
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_2 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_2.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_2))
        print("finished test with ration parameter:", ratio)

        ratio = 3
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_3 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_3.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_3))
        print("finished test with ration parameter:", ratio)

        ratio = 4
        print("starting test_resize_vocoder with ratio:", ratio)
        wav_data_4 = sol2.resize_vocoder(wav_data_orig[1], ratio)
        self.assertEqual(wav_data_orig[1].dtype, wav_data_4.dtype)
        self.assertEqual(calculate_spectogram_size(len(wav_data_orig[1]), ratio), len(wav_data_4))
        print("finished test with ration parameter:", ratio)

    def test_conv_der(self):
        filename = r"monkey.jpg"
        even_even_image = rgb2gray(imread(filename))
        result_even_even = sol2.conv_der(even_even_image)
        self.assertEqual(even_even_image.shape, result_even_even.shape)
        self.assertEqual(even_even_image.dtype, result_even_even.dtype)

        filename = r"even_odd.jpg"
        even_odd_image = rgb2gray(imread(filename))
        result_even_odd = sol2.conv_der(even_odd_image)
        self.assertEqual(even_odd_image.shape, result_even_odd.shape)
        self.assertEqual(even_odd_image.dtype, result_even_odd.dtype)

        filename = r"odd_even.jpg"
        odd_even_image = rgb2gray(imread(filename))
        result_odd_even = sol2.conv_der(odd_even_image)
        self.assertEqual(odd_even_image.shape, result_odd_even.shape)
        self.assertEqual(odd_even_image.dtype, result_odd_even.dtype)

        filename = r"odd_odd.jpg"
        odd_odd_image = rgb2gray(imread(filename))
        result_odd_odd = sol2.conv_der(odd_odd_image)
        self.assertEqual(odd_odd_image.shape, result_odd_odd.shape)
        self.assertEqual(odd_odd_image.dtype, result_odd_odd.dtype)

    def test_fourier_der(self):
        print("Starting test_fourier_der with monkey.jpg picture")
        filename = r"monkey.jpg"
        even_even_image = rgb2gray(imread(filename))
        result_even_even = sol2.fourier_der(even_even_image)
        self.assertEqual(even_even_image.shape, result_even_even.shape)
        self.assertEqual(even_even_image.dtype, result_even_even.dtype)
        print("Finished test_fourier_der with monkey.jpg picture")

        print("Starting test_fourier_der with even_odd.jpg picture")
        filename = r"even_odd.jpg"
        even_odd_image = rgb2gray(imread(filename))
        result_even_odd = sol2.fourier_der(even_odd_image)
        self.assertEqual(even_odd_image.shape, result_even_odd.shape)
        self.assertEqual(even_odd_image.dtype, result_even_odd.dtype)
        print("Finished test_fourier_der with even_odd.jpg picture")

        print("Starting test_fourier_der with odd_even.jpg picture")
        filename = r"odd_even.jpg"
        odd_even_image = rgb2gray(imread(filename))
        result_odd_even = sol2.fourier_der(odd_even_image)
        self.assertEqual(odd_even_image.shape, result_odd_even.shape)
        self.assertEqual(odd_even_image.dtype, result_odd_even.dtype)
        print("Finished test_fourier_der with odd_even.jpg picture")

        print("Starting test_fourier_der with odd_odd.jpg picture")
        filename = r"odd_odd.jpg"
        odd_odd_image = rgb2gray(imread(filename))
        result_odd_odd = sol2.fourier_der(odd_odd_image)
        self.assertEqual(odd_odd_image.shape, result_odd_odd.shape)
        self.assertEqual(odd_odd_image.dtype, result_odd_odd.dtype)
        print("Finished test_fourier_der with odd_odd.jpg picture")


def calculate_spectogram_size(orig_size, ratio):
    stft_column_count = (orig_size / 160) - 3
    after_resize = int(stft_column_count / ratio)
    after_istft = 640 + 160 * (after_resize - 1)
    return after_istft


def show_conv_and_four_der():
    filename = r"monkey.jpg"
    plt.imshow(sol2.conv_der(rgb2gray(imread(filename))), cmap='gray')
    plt.show()
    plt.imshow(sol2.fourier_der(rgb2gray(imread(filename))), cmap='gray')
    plt.show()


if __name__ == '__main__':
    unittest.main()
    # show_conv_and_four_der()

