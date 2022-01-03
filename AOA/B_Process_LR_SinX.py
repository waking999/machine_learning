from LinearRegressionSinX import LinearRegressionSinX
from AOA.generic.B_Process_LR_Generic import Process

if __name__ == '__main__':
    linear_regression_instance = LinearRegressionSinX()
    process = Process(base_file_name='E0a.txt_shuffle.csv',
                      dataset_index=0, wb_file_suffix='lr_sinx_wb.csv',
                      linear_regression_instance=linear_regression_instance,
                      learning_rate=0.0000001, eps=0.9)
    process.process()
