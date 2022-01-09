from AOA.generic.LinearRegressionGenericMultiple import LinearRegression


class LinearRegressionX(LinearRegression):
    def __init__(self, base_file_name, theta_file_suffix, num_of_variables):
        super().__init__(base_file_name=base_file_name, theta_file_suffix=theta_file_suffix,
                         num_of_variables=num_of_variables)
        return

    @staticmethod
    def origin_function(x):
        return x
