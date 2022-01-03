from AOA.generic.C_Predict_NLR_Generic import PredictNLR

if __name__ == '__main__':
    predict = PredictNLR(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='gbdt',curve_step=10000)
    predict.valuation()
