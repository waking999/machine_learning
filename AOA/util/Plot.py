import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
# from scipy.interpolate import make_interp_spline

from AOA.util.Constants import Constants


class Plot:

    @staticmethod
    def plot(train_x, train_y, val_x, val_y, pred_val_y, curve_x, curve_y, curve_step, base_file_name, image_name):
        Plot.plot_train_val(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                            pred_val_y=pred_val_y)

        plt.plot(curve_x, curve_y, c='b')

        # x_y_cubic = interp1d(curve_x, curve_y, kind='cubic')
        # _x1 = np.linspace(min(curve_x), max(curve_x), curve_step)
        # _y1 = x_y_cubic(_x1)
        # plt.plot(_x1, _y1)

        # x_y_spline = make_interp_spline(curve_x, curve_y)
        # _x2 = np.linspace(min(curve_x), max(curve_x), curve_step)
        # _y2 = x_y_spline(_x2)
        # plt.plot(_x2, _y2)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/../' + Constants.DIRECTORY_OUTPUT + '/' + base_file_name + '_' + image_name
        plt.savefig(output_file_path)

        plt.show()

    @staticmethod
    def plot_train_val(train_x, train_y, val_x, val_y, pred_val_y):
        plt.scatter(train_x, train_y, marker='o', c='black')
        plt.scatter(val_x, val_y, marker='o', c='black')
        plt.scatter(val_x, pred_val_y, marker='*', c='red')

