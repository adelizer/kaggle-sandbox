"""
A visualization script to plot surfaces for the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal


class Visualizer(object):
    def __init__(self):
        self.x_train_df = pd.read_csv('../data/X_train.csv')
        self.y_train_df = pd.read_csv('../data/y_train.csv')
        self.df = self.x_train_df.merge(self.y_train_df, on=['series_id'], how='left')
        self.surfaces = self.y_train_df.surface.unique()
        self.surface_counts = self.y_train_df.surface.value_counts().values
        self.cumsum = np.cumsum(self.surface_counts)
        self.cumsum = np.insert(self.cumsum, 0, 0)

    def my_welch(self, x):
        _, out = signal.welch(x, nperseg=128)
        return out

    def visualize_3d(self):
        x_ax = np.arange(0, 65)
        m = self.df.series_id.nunique()
        self.df = self.df.sort_values(by=['surface'])
        self.df = self.df.drop(['measurement_number', 'row_id', 'surface'], axis=1)
        values = self.df.values
        values = np.reshape(values, (m, 128, len(self.df.columns)))
        values = np.apply_along_axis(self.my_welch, 1, values)
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        print(self.cumsum)

        for i in range(len(self.cumsum)-1):
            y_ax = np.arange(self.cumsum[i], self.cumsum[i+1])
            X, Y = np.meshgrid(x_ax, y_ax)

            surf = ax.plot_surface(X, Y, values[self.cumsum[i]:self.cumsum[i+1], :, -7], label=self.surfaces[i])

            # https://github.com/matplotlib/matplotlib/issues/4067
            surf._facecolors2d = surf._facecolors3d
            surf._edgecolors2d = surf._edgecolors3d
        ax.set_xlabel("freq")
        ax.legend()
        plt.show()


def main():
    vis = Visualizer()
    vis.visualize_3d()


if __name__ == '__main__':
    main()