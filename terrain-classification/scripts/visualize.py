"""
A visualization script to plot surfaces for the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal


def my_welch(x):
    _, out = signal.welch(x, nperseg=128)
    return out


def visualize_3d(df, surface_count, surfaces):
    x_ax = np.arange(0, 65)
    m = df.series_id.nunique()

    df = df.sort_values(by=['surface'])
    df = df.drop(['measurement_number', 'row_id', 'surface'], axis=1)
    values = df.values
    values = np.reshape(values, (m, 128, len(df.columns)))
    values = np.apply_along_axis(my_welch, 1, values)
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    for i in range(len(surface_count)-1):
        y_ax = np.arange(surface_count[i], surface_count[i+1])
        X, Y = np.meshgrid(x_ax, y_ax)

        surf = ax.plot_surface(X, Y, values[surface_count[i]:surface_count[i+1], :, -7], label=surfaces[i])

        # https://github.com/matplotlib/matplotlib/issues/4067
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    ax.set_xlabel("freq")
    ax.legend()


def main():
    x_train_df = pd.read_csv('../data/X_train.csv')
    y_train_df = pd.read_csv('../data/y_train.csv')
    surfaces = y_train_df.surface.unique()

    surface_count = np.zeros((10), dtype=int)
    for i in range(y_train_df.surface.nunique()):
        surface_count[i+1] = (y_train_df.loc[y_train_df.surface == surfaces[i]].surface.count()) + surface_count[i]


    all_df = x_train_df.merge(y_train_df, on=['series_id'], how='left')
    visualize_3d(all_df, surface_count, surfaces)
    plt.show()


if __name__ == '__main__':
    main()