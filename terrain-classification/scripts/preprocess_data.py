"""
prepares the data
"""
import cesium
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from cesium import featurize


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
        curr = df.loc[df.series_id == i, 'angular_velocity_X':'linear_acceleration_Z'].values #(128, 10)
        for j in range(curr.shape[1]):
            sig = curr[:, j]
            _, out_welch = signal.welch(sig, nperseg=128)
            out_fft = np.fft.fft(sig)
            welch_array[i,j,:] = out_welch
            fft_array[i,j,:] = out_fft[1:]

    return welch_array, fft_array


def gen_feats(data):
    features_to_use = cesium.features.GENERAL_FEATS
    feats_to_remove = ['amplitude',
        'flux_percentile_ratio_mid20',
 'flux_percentile_ratio_mid35',
 'flux_percentile_ratio_mid50',
 'flux_percentile_ratio_mid65',
 'flux_percentile_ratio_mid80',
 'max_slope',
 # 'maximum',
 'median',
 'median_absolute_deviation',
 'minimum',
 'percent_amplitude',
 'percent_beyond_1_std',
 'percent_close_to_median',
 'percent_difference_flux_percentile',
 'period_fast',
 'qso_log_chi2_qsonu',
 'qso_log_chi2nuNULL_chi2nu',
 'skew',
 'std',
 'stetson_j',
 'stetson_k',
 'weighted_average']
    for f in feats_to_remove:
        features_to_use.remove(f)
    print(features_to_use)

    m = data.shape[0]

    n_features = len(features_to_use) * data.shape[1]

    agg = np.empty((m, n_features))
    for i in tqdm(range(m)):  # for all training examples
        curr = data[i, :, :]
        fset_curr = featurize.featurize_time_series(times=None,
                                                    values=curr,
                                                    # values: (n,) array or (p, n) array (for p channels of measurement)
                                                    errors=None,
                                                    features_to_use=features_to_use)
        agg[i, :] = fset_curr
    return agg


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
    # x_train_df, y_train_df = read_data()
    # welch_train = extract_welch(x_train_df)
    welch_train = np.load('welch_train.npy')
    print("Extracted welch response from training data: ", welch_train.shape)
    welch_train_feats = gen_feats(welch_train[:,0:2,:])
    print("Extracted features from welch response: ", welch_train_feats.shape)
    np.save('welch_train', welch_train)
    np.save('welch_train_feats', welch_train_feats)


if __name__ == '__main__':
    main()