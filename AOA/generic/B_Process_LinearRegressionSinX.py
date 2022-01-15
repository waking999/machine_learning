import os
import numpy as np

from AOA.util.DataSet import DataSet
from AOA.util.Constants import Constants
from AOA.util.Util import Util


class Process:
    def __init__(self, base_file_name, dataset_index, sinx_file_suffix, linear_regression_instance, learning_rate, eps):
        self.base_file_name = base_file_name
        self.dataset_index = dataset_index
        self.sinx_file_suffix = sinx_file_suffix
        self.linear_regression_instance = linear_regression_instance
        self.learning_rate = learning_rate
        self.eps = eps
        return

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               dataset_index=self.dataset_index)

        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               dataset_index=self.dataset_index)

        abcd_array = self.linear_regression_instance.get_abcd_by_sgd(xs=train_x, ys=train_y,
                                                                     learning_rate=self.learning_rate,
                                                                     a_start=0, b_start=0, c_start=0, d_start=0,
                                                                     eps=self.eps)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + self.sinx_file_suffix
        np.savetxt(output_file_path, abcd_array, delimiter=',',
                   fmt='%.' + str(Util.get_decimal_length(abcd_array[0])) + 'f')
