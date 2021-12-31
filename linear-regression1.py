import numpy as np
import os

def compute_error_for_line_given_points(b, w, points):
    total_error = 0
    for point in points:
        x = point[0]
        y = point[1]
        total_error += (y - (w * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    len_points = float(len(points))
    for point in points:
        x = point[0]
        y = point[1]
        b_gradient += ((w_current * x + b_current) - y)
        w_gradient += x * ((w_current * x + b_current) - y)

    new_b = b_current - (learning_rate * b_gradient * (2 / len_points))
    new_w = w_current - (learning_rate * w_gradient * (2 / len_points))
    return [new_b, new_w]


def gradient_descent_runner(points, b_start, w_start, learning_rate, num_iteration):
    b = b_start
    w = w_start
    for i in range(num_iteration):
        b, w = step_gradient(b, w, points, learning_rate)
    return [b, w]


def run():
    path = os.path.dirname(__file__)

    points = np.array(np.genfromtxt(path+"/ch02-data.csv", delimiter=","))
    learning_rate = 0.0003
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 300
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()