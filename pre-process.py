import glob
import librosa
import numpy as np
import pandas
import numpy
from keras_preprocessing import sequence
from librosa.feature import delta
from scipy.fft import dct

sample_rate = 16000
pre_emphasis = 0.95
frame_size = 0.08
frame_stride = 0.04
NFFT = 512
nfilt = 26
num_ceps = 13


def f_bank(signal):
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(
        float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal,
                              z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(frame_length)

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    #mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    #mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    return filter_banks


def extract_features(file):
    signal, sr = librosa.load(file, sr=sample_rate, mono=False)
    #signal = numpy.asfortranarray(numpy.concatenate([[signal[0]], [signal[1]]]))
    #signal = librosa.to_mono(signal)
    signal = signal[1]
    filter_banks = f_bank(signal)
    d = delta(filter_banks, order=1)
    d2 = delta(d, order=2)
    S = numpy.concatenate([filter_banks, d, d2], axis=1)

    return S


def test_data_generator(train_label_path, train_audio_path):
    data = []
    chew_sum = 0
    sw_sum = 0
    label = []
    for i in range(len(train_audio_path)):
        print("===", i, "===")
        print(train_audio_path[i])
        print(train_label_path[i])
        print("=========")
        tempLabel = []
        f = open(train_label_path[i], "r")
        for x in f:
            tempLabel.append(0)
            temp = x.split()
            if temp[2] == 'swallow':
                tempLabel.append(2)
                sw_sum = sw_sum + 1
            else:
                tempLabel.append(1)
                chew_sum = chew_sum + 1
        tempLabel.append(0)
        feat = extract_features(train_audio_path[i])
        data.append(feat)
        label.append(tempLabel)

    t = []
    for i in range(len(data)):
        t.append(data[i].shape[0])
    maxlen = max(t)
    data = sequence.pad_sequences(data, maxlen=maxlen)

    nb_labels = 3
    label = sequence.pad_sequences(label, value=int(nb_labels), dtype='int32', padding="post")

    print("chewsum", chew_sum)
    print("swallowsum", sw_sum)

    return data, label


def train_data_generator():
    data = []
    chew_sum = 0
    sw_sum = 0
    label = []

    train_label_path = sorted(glob.glob('data/train_data_cross_validation/*/*.txt'))
    train_audio_path = sorted(glob.glob('data/train_data_cross_validation/*/*.wav'))
    for i in range(len(train_audio_path)):
        print("===", i, "===")
        print(train_audio_path[i])
        print(train_label_path[i])
        print("=========")
        tempLabel = []
        f = open(train_label_path[i], "r")
        for x in f:
            tempLabel.append(0)
            temp = x.split()
            if temp[2] == 's':
                tempLabel.append(2)
                sw_sum = sw_sum + 1
            else:
                tempLabel.append(1)
                chew_sum = chew_sum + 1
        tempLabel.append(0)
        feat = extract_features(train_audio_path[i])
        data.append(feat)
        label.append(tempLabel)


    t = []
    for i in range(len(data)):
        t.append(data[i].shape[0])
    maxlen = max(t)
    data = sequence.pad_sequences(data, maxlen=maxlen)

    nb_labels = 3
    label = sequence.pad_sequences(label, value=int(nb_labels), dtype='int32', padding="post")

    print("chewsum", chew_sum)
    print("swallowsum", sw_sum)

    return data, label


test_label_path = sorted(glob.glob('file/ritz/*.txt'))
test_audio_path = sorted(glob.glob('file/ritz/*.wav'))
X_test, Y_test = test_data_generator(test_label_path, test_audio_path)
np.save('tempData/rtzx.npy', np.expand_dims(X_test, -1)+1.3)
np.save('tempData/rtzy.npy', Y_test.astype(np.int))

X_train, Y_train = train_data_generator()
np.save('tempData/xtrain10.npy', np.expand_dims(X_train, -1)+1.3)
np.save('tempData/ytrain10.npy', Y_train.astype(np.int))

'''print(X_train[0].shape)
print(labels[0].shape)
print(input_length[0])
print(label_length[0])'''



