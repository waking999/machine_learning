from AOA.generic.LinearRegressionGeneric import LinearRegression


class LinearRegressionX(LinearRegression):
    def __init__(self, base_file_name, wb_file_suffix):
        super().__init__(base_file_name=base_file_name, wb_file_suffix=wb_file_suffix)
        return

    @staticmethod
    def origin_function(x):
        return x
