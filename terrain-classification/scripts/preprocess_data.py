"""
prepares the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm


def read_data():
    x_train_df = pd.read_csv('../data/X_train.csv')
    y_train_df = pd.read_csv('../data/y_train.csv')
    return x_train_df, y_train_df


def compute_freq(df):
    m = df.series_id.nunique()
    initial_sample = df.loc[df.series_id==0,'orientation_X'].values
    _, temp_welch = signal.welch(initial_sample, nperseg=128)
    n_welch = temp_welch.shape[0]
    temp_fft = np.fft.fft(initial_sample)
    n_fft = temp_fft.shape[0]-1

    welch_array = np.empty((m, 10, n_welch))
    fft_array = np.empty((m, 10, n_fft))

    for i in tqdm(range(m)):
        curr = df.loc[df.series_id == i, 'orientation_X':'linear_acceleration_Z'].values #(128, 10)
        for j in range(curr.shape[1]):
            sig = curr[:, j]
            _, out_welch = signal.welch(sig, nperseg=128)
            out_fft = np.fft.fft(sig)
            welch_array[i,j,:] = out_welch
            fft_array[i,j,:] = out_fft[1:]

    return welch_array, fft_array


def extract_welch(df):
    m = df.series_id.nunique()
    welch_agg = np.empty((m, 6, 33))
    for i in tqdm(range(m)):
        curr = df.loc[df.series_id == i, 'angular_velocity_X':'linear_acceleration_Z']
        for c in range(len(curr.columns)):
            _, out = signal.welch(curr.iloc[:,c].values, nperseg=64)
            welch_agg[i, c, :] = out

    return welch_agg


def main():
    x_train_df, y_train_df = read_data()
    welch_train = extract_welch(x_train_df)
    print("Extracted welch response from training data: ", welch_train.shape)
    np.save('welch_train', welch_train)


if __name__ == '__main__':
    main()