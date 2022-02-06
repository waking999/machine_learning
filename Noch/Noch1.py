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

        self.decision_tree = {"model": DecisionTreeRegressor(max_depth=12), "marker": '+', "color": 'lightpink'}
        self.knn = {"model": KNeighborsRegressor(n_neighbors=3), "marker": '+', "color": 'crimson'}
        self.svr = {"model": SVR(C=50.0, cache_size=200, degree=100, epsilon=0.001,
                                 gamma=0.1, kernel='rbf',
                                 max_iter=-1, shrinking=True, tol=0.001, verbose=False), "marker": '2',
                    "color": 'orange'}
        self.xgb = {"model": xgb.XGBRegressor(max_depth=127, learning_rate=0.01, n_estimators=1000,
                                              objective='reg:tweedie', n_jobs=-1, booster='gbtree'), "marker": '3',
                    "color": 'darkorange'}
        self.lr = {"model": LinearRegression(), "marker": '4', "color": 'firebrick'}
        self.rf = {"model": RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), "marker": '8',
                   "color": 'tomato'}

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        _df = pd.read_csv(file_path, sep=sep_char, header=header)
        return _df

    def err(self, p, x, y):
        return p[0] * x + p[1] - y

    # def base_plot(self, base_data_file):
    #     df = self.load_dataset(self._local_dir + base_data_file, sep_char=',', header=None)
    #
    #     x_data_base = df.loc[:, 1].to_numpy()
    #     y_data_base = df.loc[:, 2].to_numpy()
    #
    #     k, b = self.linear_fitting(x_data_base, y_data_base)
    #     y_data_linearfitting = k * x_data_base + b
    #
    #     # linear fitting
    #     # plt.scatter(x_data_base, y_data_base, marker='.', c='lightgreen')
    #     plt.plot(x_data_base, y_data_linearfitting.T, c='darkred')

    def training_plot(self, training_data_file):
        # load training
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
        p0 = [100, 20]
        ret = leastsq(self.err, p0, args=(x_data, y_data))

        return ret[0]

    def base_plot(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.traing_set_base_seq * self.set_size:(self.traing_set_base_seq + 1) * self.set_size, 1]
        y_data_base = y_data_training[
                      self.traing_set_base_seq * self.set_size:(self.traing_set_base_seq + 1) * self.set_size]

        k, b = self.linear_fitting(x_data_base, y_data_base)
        y_data_linearfitting = k * x_data_base + b
        plt.plot(x_data_base, y_data_linearfitting.T, c='darkred')

    def model_fit(self, x_data_training, y_data_training, fix_temperature, model_dict):
        x_data_training_notemp = x_data_training.copy()
        x_data_training_notemp[:, 0] = fix_temperature
        model_dict["model"].fit(X=x_data_training, y=y_data_training)
        return model_dict

    def model_prdict(self, model_dict, x_data_test, y_data_test):
        y_data_model = model_dict['model'].predict(x_data_test)
        x_data_test_size = len(x_data_test)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.traing_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_model[i], marker='+', color=self.traing_set_plot_color[i])

    def process(self):
        x_data_training, y_data_training = self.training_plot(training_data_file='/input/noch1-training.csv')
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        plt.show()

        # load training
        test_data_file = '/input/noch1-test.csv'
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()

        # # decision_tree
        # self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        # model_dict_dt = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
        #                                fix_temperature=self.temperature_base,
        #                                model_dict=self.decision_tree)
        #
        # self.model_prdict(model_dict_dt, x_data_test, y_data_test)
        # plt.show()

        # knn
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        model_dict_knn = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                        fix_temperature=self.temperature_base,
                                        model_dict=self.knn)

        self.model_prdict(model_dict_knn, x_data_test, y_data_test)
        plt.show()

        # # svr
        # self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        # model_dict_svr = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
        #                                 fix_temperature=self.temperature_base,
        #                                 model_dict=self.svr)
        #
        # self.model_prdict(model_dict_svr, x_data_test, y_data_test)
        # plt.show()

        # # lr
        # self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        # model_dict_lr = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
        #                                 fix_temperature=self.temperature_base,
        #                                 model_dict=self.lr)
        #
        # self.model_prdict(model_dict_lr, x_data_test, y_data_test)
        # plt.show()

        # xgb
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        model_dict_xgb = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                        fix_temperature=self.temperature_base,
                                        model_dict=self.xgb)

        self.model_prdict(model_dict_xgb, x_data_test, y_data_test)
        plt.show()


        # rf
        self.base_plot(x_data_training=x_data_training, y_data_training=y_data_training)
        model_dict_rf = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                        fix_temperature=self.temperature_base,
                                        model_dict=self.rf)

        self.model_prdict(model_dict_rf, x_data_test, y_data_test)
        plt.show()


if __name__ == '__main__':
    noch = Noch1()
    noch.process()
