import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Noch1:
    def __init__(self):
        self.num_training_set = 5
        self.traing_set_base_seq = 2
        self.temperature_base = 30.9
        self.traing_set_plot_color = ['red', 'gold', 'green', 'black', 'purple']
        self.set_size = 14
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.models = [
            DecisionTreeRegressor(max_depth=12),
            KNeighborsRegressor(n_neighbors=3),
            # SVR(C=50.0, cache_size=200, degree=100, epsilon=0.001,
            #     gamma=0.1, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # # xgb.XGBRegressor(max_depth=127, learning_rate=0.001, n_estimators=1000,
            # #                  objective='reg:tweedie', n_jobs=-1, booster='gbtree'),
            # LinearRegression(),
            RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        ]

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        _df = pd.read_csv(file_path, sep=sep_char, header=header)
        return _df

    def err(self, p, x, y):
        return p[0] * x + p[1] - y

    def training_plot(self, training_data_file):
        df = self.load_dataset(self._local_dir + training_data_file, sep_char=',', header=None)
        x_data_training = df.loc[:, 0:1].to_numpy()
        y_data_training = df.loc[:, 2].to_numpy()

        for i in range(self.num_training_set):
            x_data_training_tmp = x_data_training[i * self.set_size:(i + 1) * self.set_size, 1]
            y_data_training_tmp = y_data_training[i * self.set_size:(i + 1) * self.set_size]
            plt.scatter(x_data_training_tmp, y_data_training_tmp, marker="." if i != self.traing_set_base_seq else "x",
                        c=self.traing_set_plot_color[i])

        return x_data_training, y_data_training

    def linear_fitting(self, x_data, y_data):
        p0 = np.array([100, 20])
        x_data = x_data.astype('float')
        y_data = y_data.astype('float')
        ret = leastsq(self.err, p0, args=(x_data, y_data))
        print(ret)
        return ret[0]

    def base_plot(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.traing_set_base_seq * self.set_size:(self.traing_set_base_seq + 1) * self.set_size, 1]
        y_data_base = y_data_training[
                      self.traing_set_base_seq * self.set_size:(self.traing_set_base_seq + 1) * self.set_size]

        k, b = self.linear_fitting(x_data_base, y_data_base)
        y_data_lf = k * x_data_base + b
        plt.plot(x_data_base, y_data_lf.T, c='darkred')

    def model_fit(self, x_data_training, y_data_training, fix_temperature, model):
        x_data_training_notemp = x_data_training.copy()
        x_data_training_notemp[:, 0] = fix_temperature
        model.fit(X=x_data_training_notemp, y=y_data_training)
        return model

    def test_plot(self, test_data_file, x_data_training, y_data_training, model_seq):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()

        model_size = len(self.models)
        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                               fix_temperature=self.temperature_base,
                               model=self.models[model_seq])
        y_data_model_temp = model.predict(x_data_test)

        x_data_test_size = len(x_data_test)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.traing_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_model_temp[i], marker='+', color=self.traing_set_plot_color[i])

    def test_lf_plot(self, test_data_file, x_data_training, y_data_training):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()

        x_data_test_size = len(x_data_test)
        model_size = len(self.models)

        x_data_test_model = np.array([[None, None]])
        y_data_test_model = np.array([None])
        for i in range(model_size):
            x_data_test_model = np.vstack((x_data_test_model, x_data_training))
            model_temp = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                        fix_temperature=self.temperature_base,
                                        model=self.models[i])
            y_data_model_temp = model_temp.predict(x_data_training)
            y_data_test_model = np.hstack((y_data_test_model, y_data_model_temp))

        k, b = self.linear_fitting(x_data_test_model[1:, 1], y_data_test_model[1:])

        for i in range(x_data_test_size):
            y_data_test_lf_predict = k * x_data_test[i, 1] + b
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.traing_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_test_lf_predict, marker='+', color=self.traing_set_plot_color[i])

    def process(self):
        # training plot
        x_data_training, y_data_training = self.training_plot(training_data_file='/input/noch1-training.csv')
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        plt.show()

        # test plot
        for i in range(len(self.models)):
            self.test_plot(test_data_file='/input/noch1-test.csv', x_data_training=x_data_training,
                           y_data_training=y_data_training, model_seq=i)
            self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
            plt.show()

        # test lf plot
        self.test_lf_plot(test_data_file='/input/noch1-test.csv', x_data_training=x_data_training,
                          y_data_training=y_data_training)
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        plt.show()


if __name__ == '__main__':
    noch = Noch1()
    noch.process()
