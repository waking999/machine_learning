import os
import numpy as np

from DataSet import DataSet
from LinearRegression import LinearRegression
from Util import Util


class Process:
    def __init__(self):
        return

    @staticmethod
    def process(base_file_name, index, learning_rate, eps):
        _local_dir = os.path.dirname(__file__)

        input_file_path_train_x = _local_dir + '/input/' + base_file_name + '_' + 'train_x' + str(index) + '.csv'
        df_train_x = DataSet.load_dataset(file_path=input_file_path_train_x, sep_char=',', header=None)
        train_x = df_train_x[0]

        input_file_path_train_y = _local_dir + '/input/' + base_file_name + '_' + 'train_y' + str(index) + '.csv'
        df_train_y = DataSet.load_dataset(file_path=input_file_path_train_y, sep_char=',', header=None)
        train_y = df_train_y[0]

        wb_array = LinearRegression.get_w_b_by_sgd(xs=train_x, ys=train_y,
                                                   learning_rate=learning_rate,
                                                   w_start=0, b_start=0, eps=eps)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/output/' + base_file_name + '_' + 'wb' + '.csv'
        np.savetxt(output_file_path, wb_array, delimiter=',', fmt='%.' + Util.get_decimal_length(wb_array[0]) + 'f')


if __name__ == '__main__':
    Process.process(base_file_name='E0a.txt_shuffle.csv', index=0, learning_rate=0.00001, eps=0.01)
