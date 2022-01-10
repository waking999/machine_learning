from AOA.generic.LinearRegressionSinX import LinearRegressionSinX
from AOA.generic.C_Predict_LinearRegressionSinX import PredictLR

if __name__ == '__main__':
    base_file_name = 'E0a.txt_shuffle.csv'
    sinx_file_suffix = 'lr_sinx_abcd.csv'
    linear_regression_instance = LinearRegressionSinX(base_file_name=base_file_name,
                                                      sinx_file_suffix=sinx_file_suffix)
    predict = PredictLR(base_file_name=base_file_name, dataset_index=0, sinx_file_suffix=sinx_file_suffix,
                        linear_regression_instance=linear_regression_instance, curve_step=10000)
    predict.valuation()
