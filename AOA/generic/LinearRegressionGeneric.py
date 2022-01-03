class LinearRegression:
    def __init__(self):
        return

    def origin_function(self, x):
        return x

    def compute_error_for_line_given_points(self, xs, ys, w, b):
        total_error = 0
        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            total_error += (y - (w * self.origin_function(x) + b)) ** 2

        return total_error / float(len_xs)

    def step_gradient(self, xs, ys, w_current, b_current, learning_rate):
        b_gradient = 0
        w_gradient = 0
        c_gradient = 1
        len_xs = len(xs)
        for i in range(len_xs):
            x = xs[i]
            y = ys[i]
            b_gradient += ((w_current * self.origin_function(x) + b_current) - y)
            w_gradient += self.origin_function(x) * ((w_current * self.origin_function(x) + b_current) - y)

        new_w = w_current - (learning_rate * w_gradient * (2 / float(len_xs)))
        new_b = b_current - (learning_rate * b_gradient * (2 / float(len_xs)))

        return [new_w, new_b]

    def gradient_descent_runner(self, xs, ys, w_start, b_start, learning_rate, eps):
        b = b_start
        w = w_start

        error_before = self.compute_error_for_line_given_points(xs=xs, ys=ys, w=w_start, b=b_start)
        print("Before Linear Regression at w = {0}, b = {1}, error = {2}"
              .format(w_start, b_start, error_before)
              )

        print("Running...")
        i = 0
        error_after = error_before
        error_diff_rate = ((error_before - error_after) / error_before)
        while error_diff_rate < (1 - eps):
            w, b = self.step_gradient(xs=xs, ys=ys, w_current=w, b_current=b, learning_rate=learning_rate)
            error_after = self.compute_error_for_line_given_points(xs=xs, ys=ys, w=w, b=b)
            error_diff_rate = ((error_before - error_after) / error_before)
            print("After {0} iterations w = {1}, b = {2}, error = {3}, real_time_eps={4}".
                  format(i, w, b, error_after, (1 - error_diff_rate))
                  )
            i += 1

        return [w, b]

    def get_w_b_by_sgd(self, xs, ys, learning_rate, w_start, b_start, eps):

        [w, b] = self.gradient_descent_runner(xs=xs, ys=ys,
                                              w_start=w_start, b_start=b_start,
                                              learning_rate=learning_rate, eps=eps)

        return [w, b]

    def predict(self, xs, w, b):
        y_pred = []
        for x in xs:
            tmp_y = w * self.origin_function(x) + b
            y_pred.append(tmp_y)
        return y_pred
