import math
from AOA.generic.LinearRegressionGeneric import LinearRegression


class LinearRegressionSinX(LinearRegression):
    def __init__(self):
        super().__init__()
        return

    def origin_function(self, x):
        return math.sin(x)
