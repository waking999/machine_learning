import os
import numpy as np

from AOA.util.DataSet import DataSet
from AOA.util.Constants import Constants
from AOA.util.Util import Util


class Process:
    def __init__(self, base_file_name, dataset_index, theta_file_suffix, num_of_variables, linear_regression_instance,
                 learning_rate, eps):
        self.base_file_name = base_file_name
        self.dataset_index = dataset_index
        self.theta_file_suffix = theta_file_suffix
        self.linear_regression_instance = linear_regression_instance
        self.learning_rate = learning_rate
        self.eps = eps
        self.num_of_variables = num_of_variables
        return

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               dataset_index=self.dataset_index)

        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               dataset_index=self.dataset_index)

        theta_start = np.zeros(self.num_of_variables)
        theta = self.linear_regression_instance.get_theta_by_sgd(xs=train_x, ys=train_y,
                                                                 learning_rate=self.learning_rate,
                                                                 theta_start=theta_start, eps=self.eps)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + str(self.num_of_variables) + '_' + self.theta_file_suffix
        np.savetxt(output_file_path, X=theta, delimiter=',',
                   fmt='%.' + str(Util.get_decimal_length(theta[0])) + 'f')
