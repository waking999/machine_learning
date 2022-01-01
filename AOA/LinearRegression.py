import numpy as np


class LinearRegression:
    def __init__(self):
        return

    @staticmethod
    def compute_error_for_line_given_points(xs, ys, w, b):
        total_error = 0
        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            total_error += (y - (w * x + b)) ** 2

        return total_error / float(len_xs)

    @staticmethod
    def step_gradient(xs, ys, w_current, b_current, learning_rate):
        b_gradient = 0
        w_gradient = 0
        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            b_gradient += ((w_current * x + b_current) - y)
            w_gradient += x * ((w_current * x + b_current) - y)

        new_w = w_current - (learning_rate * w_gradient * (2 / float(len_xs)))
        new_b = b_current - (learning_rate * b_gradient * (2 / float(len_xs)))

        return [new_w, new_b]

    @staticmethod
    def gradient_descent_runner(xs, ys, w_start, b_start, learning_rate, eps):
        b = b_start
        w = w_start

        error_before = LinearRegression.compute_error_for_line_given_points(xs=xs, ys=ys, w=w_start, b=b_start)
        i = 0
        error_after = error_before
        while np.absolute(error_before - error_after) / error_before < 1 - eps:
            w, b = LinearRegression.step_gradient(xs=xs, ys=ys, w_current=w, b_current=b, learning_rate=learning_rate)
            error_after = LinearRegression.compute_error_for_line_given_points(xs=xs, ys=ys, w=w, b=b)
            print("After {0} iterations w = {1}, b = {2}, error = {3}".
                  format(i, w, b, error_after)
                  )
            i += 1

        return [i, w, b]

    @staticmethod
    def get_w_b_by_sgd(xs, ys, learning_rate, w_start, b_start, eps):
        error_before = LinearRegression.compute_error_for_line_given_points(xs=xs, ys=ys, w=w_start, b=b_start)
        print("Before Linear Regression at w = {0}, b = {1}, error = {2}"
              .format(w_start, b_start, error_before)
              )

        print("Running...")
        [num_iterations, w, b] = LinearRegression.gradient_descent_runner(xs=xs, ys=ys,
                                                                          w_start=w_start, b_start=b_start,
                                                                          learning_rate=learning_rate, eps=eps)

        error_after = LinearRegression.compute_error_for_line_given_points(xs=xs, ys=ys, w=w, b=b)
        print("After {0} iterations w = {1}, b = {2}, error = {3}".
              format(num_iterations, w, b, error_after)
              )

        return [w, b]

    @staticmethod
    def predict(xs, w, b):
        y_pred = []
        for x in xs:
            tmp_y = w * x + b
            y_pred.append(tmp_y)
        return y_pred
