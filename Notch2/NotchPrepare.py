import os
import numpy as np
import pandas as pd

from scipy.optimize import leastsq
from sklearn.metrics import r2_score


# calculate k, b

class NotchPrepare:
    def __init__(self):
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.num_training_set = 7
        self.set_size = 9
        self.training_set_base_seq = self.num_training_set // 2

        self.k = None
        self.b = None

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        _df = pd.read_csv(file_path, sep=sep_char, header=header)
        return _df

    def load_data(self, data_file):
        df = self.load_dataset(self._local_dir + data_file, sep_char=',', header=None)
        x_data = df.loc[:, 1].to_numpy()
        y_data = df.loc[:, [2, 3]].to_numpy()
        return x_data, y_data, df

    def err(self, p, x, y):
        return p[0] * x + p[1] - y

    def linear_fitting(self, x_data, y_data):
        p0 = np.array([100, 20])
        x_data = x_data.astype('float')
        y_data = y_data.astype('float')
        ret = leastsq(self.err, p0, args=(x_data, y_data))

        y_data_lr = ret[0][0] * x_data + ret[0][1]
        print('R2:' + str(r2_score(y_data, y_data_lr)))
        print('k,b=' + str(ret))
        return ret[0]

    def calculate_lf_kb(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        y_data_base = y_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size, 0]
        k, b = self.linear_fitting(x_data_base, y_data_base)
        return k, b 

    def proces(self):
        # load training set, take the base to get expectation
        data_file_training = './input/notch-training.csv'
        x_data_training, y_data_training, df_training = self.load_data(data_file=data_file_training)
        # calculate k,b
        lf_k, lf_b = self.calculate_lf_kb(x_data_training, y_data_training)  
        # save training with expectation
        y_data_training_base = lf_k * x_data_training[:] + lf_b
        df_training[4] = y_data_training_base
        df_training.to_csv("./input/notch-training-afterlf.csv", index=False, header=False)

        # save validation with expectation
        data_file_validation = './input/notch-validation.csv'
        x_data_validation, y_data_validation, df_validation = self.load_data(data_file=data_file_validation)
        data_tag_validation = lf_k * x_data_validation[:] + lf_b
        df_validation[4] = data_tag_validation
        df_validation.to_csv("./input/notch-validation-afterlf.csv", index=False, header=False)

        # # save test with expectation
        # data_file_test = './input/notch-test.csv'
        # x_data_test, y_data_test, df_test = self.load_data(data_file=data_file_test)
        # y_data_test_base = lf_k * x_data_test[:] + lf_b
        # df_test[4] = y_data_test_base
        # df_test.to_csv("./input/notch-test-afterlf.csv", index=False, header=False)
        # save k,b
        kb_df = pd.DataFrame()
        kb_df["k"] = [lf_k]
        kb_df["b"] = [lf_b]
        kb_df.to_csv("./input/notch-kb.csv", index=False, header=True)

        print('Notch Prepare Finish')


if __name__ == '__main__':
    notchPrepare = NotchPrepare()

    notchPrepare.proces()
