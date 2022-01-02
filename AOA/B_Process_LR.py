import os
import numpy as np

from DataSet import DataSet
from Constants import Constants
from LinearRegression import LinearRegression
from Util import Util


class Process:
    def __init__(self):
        return

    @staticmethod
    def process(base_file_name, index, learning_rate, eps):

        train_x = DataSet.load_individual_data(base_file_name=base_file_name, array_name='train_x',
                                               index=index)

        train_y = DataSet.load_individual_data(base_file_name=base_file_name, array_name='train_y',
                                               index=index)

        wb_array = LinearRegression.get_w_b_by_sgd(xs=train_x, ys=train_y,
                                                   learning_rate=learning_rate,
                                                   w_start=0, b_start=0, eps=eps)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_WORK + '/' + base_file_name + '_lr_wb.csv'
        np.savetxt(output_file_path, wb_array, delimiter=',',
                   fmt='%.' + str(Util.get_decimal_length(wb_array[0])) + 'f')


if __name__ == '__main__':
    Process.process(base_file_name='E0a.txt_shuffle.csv', index=0, learning_rate=0.00001, eps=0.007)
