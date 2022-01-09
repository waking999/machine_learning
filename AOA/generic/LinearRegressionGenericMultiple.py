from AOA.util.Constants import Constants
from AOA.util.Util import Util
from AOA.util.DataSet import DataSet

import os
import numpy as np


class LinearRegression:

    def __init__(self, base_file_name, theta_file_suffix, num_of_variables):
        self.base_file_name = base_file_name
        self.theta_file_suffix = theta_file_suffix
        self.num_of_variables = num_of_variables
        return

    def origin_function(self, x):
        return x

    def cal_multi_variables(self, theta, x):
        tmp_y = 0
        tmp_x = 1
        for j in range(self.num_of_variables):
            tmp_y += theta[j] * tmp_x
            tmp_x *= tmp_x * self.origin_function(x)
        return tmp_y

    def compute_error_for_line_given_points(self, xs, ys, theta):
        total_error = 0
        len_xs = len(xs)

        for i in range(len_xs):
            x = xs[i]
            y = ys[i]

            tmp_y = self.cal_multi_variables(theta=theta, x=x)
            total_error += (y - tmp_y) ** 2

        return total_error / float(len_xs)

    def step_gradient(self, xs, ys, theta_current, learning_rate):

        theta_gradient = np.zeros(self.num_of_variables)

        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]

            tmp_y = 1
            for j in range(self.num_of_variables):
                theta_gradient[j] += ((self.cal_multi_variables(theta=theta_current, x=x) - y) * tmp_y)
                tmp_y *= self.origin_function(x)

        theta_new = np.zeros(self.num_of_variables)

        for i in range(self.num_of_variables):
            theta_new[i] = theta_current[i] - (learning_rate * theta_gradient[i] * (2 / float(len_xs)))

        return theta_new

    def gradient_descent_runner(self, xs, ys, theta_start, learning_rate, eps):
        try:
            _local_dir = os.path.dirname(__file__)
            input_file_path_theta = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + str(
                self.num_of_variables) + '_' + self.theta_file_suffix
            df_theta = DataSet.load_dataset(file_path=input_file_path_theta, sep_char=',', header=None)
            theta = df_theta[0]

            for i in range(self.num_of_variables):
                if theta[i] is None:
                    theta[i] = theta_start
        except FileNotFoundError:
            theta = theta_start

        error_before = self.compute_error_for_line_given_points(xs=xs, ys=ys, theta=theta)
        print(theta)
        print("error = {0}".format(error_before))

        print("Running...")
        i = 0
        error_after = error_before
        error_diff_rate = ((error_before - error_after) / error_before)
        while error_diff_rate < (1 - eps):
            theta = self.step_gradient(xs=xs, ys=ys, theta_current=theta, learning_rate=learning_rate)
            error_after = self.compute_error_for_line_given_points(xs=xs, ys=ys, theta=theta)
            error_diff_rate = ((error_before - error_after) / error_before)
            print(theta)
            print("error = {0}, real_time_eps={1}".format(error_after, (1 - error_diff_rate)))

            if error_after < error_before:
                _local_dir = os.path.dirname(__file__)
                output_file_path = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + str(
                    self.num_of_variables) + '_' + self.theta_file_suffix
                np.savetxt(fname=output_file_path, X=theta, delimiter=',',
                           fmt='%.' + str(Util.get_decimal_length(theta[0])) + 'f')
            i += 1

        return theta

    def get_theta_by_sgd(self, xs, ys, learning_rate, theta_start, eps):
        theta = self.gradient_descent_runner(xs=xs, ys=ys, theta_start=theta_start, learning_rate=learning_rate,
                                             eps=eps)
        return theta

    def predict(self, xs, theta):
        y_pred = []

        for x in xs:
            tmp_y = self.cal_multi_variables(theta=theta, x=x)
            y_pred.append(tmp_y)
        return y_pred
