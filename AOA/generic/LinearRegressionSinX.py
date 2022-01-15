import math
import os
import numpy as np

from AOA.util.Constants import Constants
from AOA.util.DataSet import DataSet
from AOA.util.Util import Util


class LinearRegressionSinX:
    def __init__(self, base_file_name, sinx_file_suffix):
        self.base_file_name = base_file_name
        self.sinx_file_suffix = sinx_file_suffix
        return

    def __compute_error_for_line_given_points(self, xs, ys, a, b, c, d):
        total_error = 0
        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            total_error += (y - (self.__sinx_generic_func(a, b, c, d, x))) ** 2

        return total_error / float(len_xs)

    def __step_gradient(self, xs, ys, a_current, b_current, c_current, d_current, learning_rate):
        a_gradient = 0
        b_gradient = 0
        c_gradient = 0
        d_gradient = 0

        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            a_gradient += (self.__sinx_generic_func(a_current, b_current, c_current, d_current, x) - y) * math.cos(
                b_current * x + c_current)
            b_gradient += (self.__sinx_generic_func(a_current, b_current, c_current, d_current, x) - y) * math.cos(
                b_current * x + c_current) * x
            c_gradient += (self.__sinx_generic_func(a_current, b_current, c_current, d_current, x) - y) * math.cos(
                b_current * x + c_current)
            d_gradient += (self.__sinx_generic_func(a_current, b_current, c_current, d_current, x) - y)

        new_a = a_current - (learning_rate * a_gradient * (2 / float(len_xs)))
        new_b = b_current - (learning_rate * b_gradient * (2 / float(len_xs)))
        new_c = c_current - (learning_rate * c_gradient * (2 / float(len_xs)))
        new_d = d_current - (learning_rate * d_gradient * (2 / float(len_xs)))

        return [new_a, new_b, new_c, new_d]

    def gradient_descent_runner(self, xs, ys, a_start, b_start, c_start, d_start, learning_rate, eps):
        _local_dir = os.path.dirname(__file__)
        input_file_path_wb = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + self.sinx_file_suffix
        df_abcd = DataSet.load_dataset(file_path=input_file_path_wb, sep_char=',', header=None)
        a = df_abcd[0][0]
        b = df_abcd[0][1]
        c = df_abcd[0][2]
        d = df_abcd[0][3]

        if a is None:
            a = a_start

        if b is None:
            b = b_start

        if c is None:
            c = c_start

        if d is None:
            d = d_start

        try:
            error_before = df_abcd[0][4]
            error_before = max(error_before, self.__compute_error_for_line_given_points(xs=xs, ys=ys, a=a, b=b, c=c, d=d))

        except KeyError:
            error_before = self.__compute_error_for_line_given_points(xs=xs, ys=ys, a=a, b=b, c=c, d=d)

        print("Before Linear Regression at a = {0}, b = {1}, c={2}, d={3}, error = {4}"
              .format(a, b, c, d, error_before)
              )

        print("Running...")
        i = 0
        error_after = error_before
        error_diff_rate = ((error_before - error_after) / error_before)
        while error_diff_rate < (1 - eps):
            abcd_array = self.__step_gradient(xs=xs, ys=ys, a_current=a, b_current=b, c_current=c, d_current=d,
                                              learning_rate=learning_rate)
            a = abcd_array[0]
            b = abcd_array[1]
            c = abcd_array[2]
            d = abcd_array[3]

            error_after = self.__compute_error_for_line_given_points(xs=xs, ys=ys, a=a, b=b, c=c, d=d)
            error_diff_rate = ((error_before - error_after) / error_before)
            print("After {0} iterations a = {1}, b = {2}, c = {3}, d = {4}, error = {5}, real_time_eps={6}".
                  format(i, a, b, c, d, error_after, (1 - error_diff_rate))
                  )
            if error_after < error_before:
                abcd_array.append(error_before)
                _local_dir = os.path.dirname(__file__)
                output_file_path = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + self.base_file_name + '_' + self.sinx_file_suffix
                np.savetxt(output_file_path, abcd_array, delimiter=',',
                           fmt='%.' + str(Util.get_decimal_length(abcd_array[0])) + 'f')
            i += 1

        return [a, b, c, d, error_before]

    def get_abcd_by_sgd(self, xs, ys, learning_rate, a_start, b_start, c_start, d_start, eps):

        return self.gradient_descent_runner(xs=xs, ys=ys,
                                            a_start=a_start, b_start=b_start, c_start=c_start, d_start=d_start,
                                            learning_rate=learning_rate, eps=eps)

    def predict(self, xs, a, b, c, d):
        y_pred = []
        for x in xs:
            tmp_y = self.__sinx_generic_func(a, b, c, d, x)
            y_pred.append(tmp_y)
        return y_pred

    @staticmethod
    def __sinx_generic_func(a, b, c, d, x):
        return a * math.sin(b * x + c) + d
