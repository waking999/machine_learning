from AOA.generic.LinearRegressionSinX import LinearRegressionSinX
from AOA.generic.B_Process_LinearRegressionSinX import Process

if __name__ == '__main__':
    base_file_name = 'E0a.txt_shuffle.csv'
    sinx_file_suffix = 'lr_sinx_abcd.csv'
    linear_regression_instance = LinearRegressionSinX(base_file_name=base_file_name,
                                                      sinx_file_suffix=sinx_file_suffix)
    process = Process(base_file_name=base_file_name,
                      dataset_index=0, sinx_file_suffix=sinx_file_suffix,
                      linear_regression_instance=linear_regression_instance,
                      learning_rate=0.00000003, eps=0.1)
    process.process()
