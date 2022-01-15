from AOA.ToBeRemoved.LinearRegressionX import LinearRegressionX
from AOA.ToBeRemoved.C_Predict_LR_Generic import PredictLR

if __name__ == '__main__':
    linear_regression_instance = LinearRegressionX(base_file_name='E0a.txt_shuffle.csv', wb_file_suffix='lr_x_wb.csv')
    predict = PredictLR(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, wb_file_suffix='lr_x_wb.csv',
                        linear_regression_instance=linear_regression_instance, curve_step=10000)
    predict.valuation()
