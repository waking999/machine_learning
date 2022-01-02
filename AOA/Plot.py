import matplotlib.pyplot as plt
import os
from Constants import Constants
import numpy as np


class Plot:
    def __init__(self):
        return

    @staticmethod
    def plot_lr(train_x, train_y, val_x, val_y, pred_val_y, base_file_name, image_name, w, b,
                line_space_x1, line_space_x2, line_space_n):
        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, pred_val_y=pred_val_y)

        x = np.linspace(line_space_x1, line_space_x2, line_space_n)
        y = w * x + b

        plt.plot(x, y, 'r')

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_OUTPUT + '/' + base_file_name + '_' + image_name
        plt.savefig(output_file_path)

        plt.show()

    @staticmethod
    def plot_knn(train_x, train_y, val_x, val_y, pred_val_y, train_x_sort, pred_train_y, base_file_name, image_name):
        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                  pred_val_y=pred_val_y)

        plt.plot(train_x_sort, pred_train_y, 'r')

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_OUTPUT + '/' + base_file_name + '_' + image_name
        plt.savefig(output_file_path)

        plt.show()

    @staticmethod
    def plot(train_x, train_y, val_x, val_y, pred_val_y):
        plt.scatter(train_x, train_y, marker='o', c='black')
        plt.scatter(val_x, val_y, marker='o', c='black')
        plt.scatter(val_x, pred_val_y, marker='*', c='red')
