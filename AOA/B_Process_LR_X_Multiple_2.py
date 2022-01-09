from LinearRegressionXMultiple import LinearRegressionX
from AOA.generic.B_Process_LR_Generic_Multiple import Process

if __name__ == '__main__':
    base_file_name = 'E0a.txt_shuffle.csv'
    theta_file_suffix = 'lr_x_theta.csv'
    num_of_variables = 2
    linear_regression_instance = LinearRegressionX(base_file_name=base_file_name,
                                                   theta_file_suffix=theta_file_suffix,
                                                   num_of_variables=num_of_variables)
    process = Process(base_file_name=base_file_name,
                      dataset_index=0, theta_file_suffix=theta_file_suffix,
                      num_of_variables=num_of_variables,
                      linear_regression_instance=linear_regression_instance,
                      learning_rate=0.00001, eps=0.00633)
    process.process()
