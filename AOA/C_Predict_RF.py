from C_Predict_Generic import Predict

if __name__ == '__main__':
    predict = Predict(base_file_name='E0a.txt_shuffle.csv', index=0, model_name='rf')
    predict.valuation()
