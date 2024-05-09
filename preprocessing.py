import pickle as pkl
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


# class to save the data as
class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# CONSTANT SR
sampling_rate = 8000


# CONSTANT FP
paths = os.listdir('./train')
paths = [os.path.join('./train', filename) for filename in paths]


# extract mfccs
def mfccfication(sound, sampling_rate=sampling_rate, n=16):
    return librosa.feature.mfcc(y=sound, sr=sampling_rate, n_mfcc=n)


# normalize mfccs
def mfcc_norm(mfccs):
    mu = np.mean(mfccs, axis=0)
    sigma = np.std(mfccs, axis=0)
    return (mfccs - mu) / sigma


# yield files
def file_generator(paths):
    for path in paths:
        yield pkl.load(open(path, 'rb'))


# combine
def tensorify(array):
    return torch.tensor(array).float()


# aligning with FP
def process_audio(audio, maxlen):
    label = audio['valence']
    mfccs = mfccfication(pad_or_trim(audio['audio_data'], maxlen))
    normalised = mfcc_norm(mfccs)
    return normalised, label


# determine the max length
def determine_max_length(data: iter):
    max_len = 0
    for item in data:
        max_len = max(max_len, len(item['audio_data']))
    return max_len


# build a dataset and save it
def build_dataset(paths):
    mfcc_list = []
    val_list = []

    maxlen = determine_max_length(mfcc_list)
    print(maxlen)

    # preprocess
    for audio in file_generator(paths):
        mfcc, valence = process_audio(audio, maxlen)
        mfcc_list.append(mfcc)
        val_list.append(valence)

    val_list = tensorify(val_list)

    assert all(entry.shape == mfcc_list[0].shape for entry in mfcc_list)

    dataset = AudioDataset(mfcc_list, val_list)
    torch.save(dataset, 'audio_dataset.pth')


# def pad_sequence(data, maxlen):
#     padding = maxlen - data.shape[1]
#     padded_array = np.pad(data, ((0, 0), (0, padding)), mode='constant')
#     return padded_array


def pad_or_trim(data, maxlen):
    if len(data) > maxlen:
        trimmed_array = np.array(data[:maxlen])
        return trimmed_array
    elif len(data) < maxlen:
        padding = maxlen - len(data)
        padded_array = np.pad(data, (0, padding), mode='constant')
        return padded_array


def inspect_shapes():
    for i in range(1, 6):
        x = pkl.load(open(f'./train/1000{i}.pkl', 'rb'))
        y = mfccfication(np.array(x['audio_data']), sampling_rate, 16)
        z = librosa.feature.melspectrogram(y=np.array(x['audio_data']), sr=sampling_rate)
        print("MFCC shape: ", np.array(y).shape)
        print("Mel Spectrogram shape: ", np.array(z).shape)


if __name__ == '__main__':
    build_dataset(paths)

    # x = np.array([[1, 3, 5], [2, 3, 5]])
    # print(x.shape, type(x.shape), x.shape[1])
    # print(pad_sequence(x, 6))
