import math
import os

# import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
# import xgboost as xgb
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class Noch2:
    def __init__(self):
        self.num_training_set = 5
        self.training_set_base_seq = 2
        self.temperature_base = 30.9
        self.training_set_plot_color = ['red', 'gold', 'green', 'black', 'purple']
        self.set_size = 14
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.models = [
            # DecisionTreeRegressor(max_depth=12),
            # KNeighborsRegressor(n_neighbors=3),
            # SVR(C=50, cache_size=200, degree=3, epsilon=0.1,
            #     gamma=2.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # SVR(C=0.005, cache_size=200, degree=3, epsilon=0.00001,
            #     gamma=0.0001, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            SVR(C=81.92, cache_size=200, degree=3, epsilon=0.00064,
                gamma=0.08192, kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001024, verbose=False)
            # # xgb.XGBRegressor(max_depth=127, learning_rate=0.001, n_estimators=1000,
            # #                  objective='reg:tweedie', n_jobs=-1, booster='gbtree'),
            # LinearRegression(),
            # RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        ]
        self.model_mae = {}
        self.favorit_svr_seq = 0

        self.model_svr_mae = pd.DataFrame()
        self.min_svr_mae = None
        self.min_svr_c = None
        self.min_svr_g = None
        self.min_svr_t = None
        self.min_svr_e = None

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
            plt.scatter(x_data_training_tmp, y_data_training_tmp,
                        marker="." if i != self.training_set_base_seq else "x",
                        c=self.training_set_plot_color[i])

        return x_data_training, y_data_training

    def set_box_color(self, bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    def linear_fitting(self, x_data, y_data):
        p0 = np.array([100, 20])
        x_data = x_data.astype('float')
        y_data = y_data.astype('float')
        ret = leastsq(self.err, p0, args=(x_data, y_data))

        y_data_lr = ret[0][0] * x_data + ret[0][1]
        print(r2_score(y_data, y_data_lr))

        print(ret)
        return ret[0]

    def base_plot(self, k, b, x_data_base):
        y_data_lf = k * x_data_base + b
        plt.plot(x_data_base, y_data_lf.T, c='darkred')

    def calculate_base_kb(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size, 1]
        y_data_base = y_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        k, b = self.linear_fitting(x_data_base, y_data_base)
        return k, b, x_data_base

    def model_fit(self, x_data_training, y_data_training, model, base_k, base_b):
        # corrected frequency
        y_data_corrected = base_k * x_data_training[:, 1] + base_b
        # measured frequencey
        y_data_measured = np.reshape(y_data_training, (-1, 1))

        model.fit(X=y_data_measured, y=y_data_corrected)
        return model

    def test_plot(self, test_data_file, x_data_training, y_data_training, model_seq, base_k, base_b):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()

        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                               base_k=base_k, base_b=base_b, model=self.models[model_seq])
        y_data_test_measured = np.reshape(y_data_test, (-1, 1))
        y_data_test_corrected = model.predict(y_data_test_measured)

        y_data_base = base_k * x_data_test[:, 1] + base_b
        # x_data_base = (y_data_test_measured - base_b) / base_k
        # x_data_corrected = (y_data_test_corrected - base_b) / base_k
        self.model_mae[model.__class__.__name__ + str(model_seq)] = mean_absolute_error(y_data_base,
                                                                                        y_data_test_corrected)

        x_data_test_size = len(x_data_test)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_test_corrected[i], marker='+', color=self.training_set_plot_color[i])

            stress_predicted = (y_data_test_corrected[i] - base_b) / base_k
            print(str(x_data_test[i, 1]) + ',' + str(y_data_test_measured[i]) + ',' + str(y_data_base[i]) + ',' + str(
                y_data_test_corrected[i]) + ',' + str(stress_predicted))

    def test_svr(self, test_data_file, x_data_training, y_data_training, base_k, base_b):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()
        y_data_base = base_k * x_data_test[:, 1] + base_b
        # x_data_base = (y_data_test - base_b)/base_k
        step = 1.5

        # parameters = [{
        #     'kernel': ['rbf'],
        #     'C': np.logspace(base=10, start=-3, stop=2, num=6, endpoint=True) * 5,
        #     'gamma': np.logspace(base=10, start=-5, stop=-1, num=5, endpoint=True),
        #     'tol': np.logspace(base=10, start=-7, stop=-1, num=7, endpoint=True),
        #     'epsilon': np.logspace(base=10, start=-5, stop=-2, num=4, endpoint=True),
        #     'max_iter': [-1],
        #     'cache_size': [200],
        #     'degree': [3],
        #     'shrinking': [True]
        # }]
        # print(parameters)

        parameters = [{
            'C': np.logspace(base=step, start=-14, stop=16, num=31, endpoint=True),
            'gamma': np.logspace(base=step, start=-29, stop=-6, num=24, endpoint=True),
            'tol': np.logspace(base=step, start=-40, stop=-6, num=35, endpoint=True),
            'epsilon': np.logspace(base=step, start=-29, stop=-12, num=18, endpoint=True)
        }]
        print(parameters)

        # clf = GridSearchCV(
        #     SVR(kernel='rbf', shrinking=True, degree=3, cache_size=200, max_iter=-1),
        #     param_grid=parameters,
        #     scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1
        # )
        # clf.fit(x_data_training, y_data_training)
        #
        # print('clf.best_params_', clf.best_params_)

        i1 = math.exp(step, -14)
        self.min_svr_mae = 1
        while i1 <= math.exp(step, 16):
            i2 = math.exp(step, -29)
            while i2 <= math.exp(step, -6):
                i3 = math.exp(step, -40)
                while i3 <= math.exp(step, -6):
                    i4 = math.exp(step, -29)
                    while i4 <= math.exp(step, -12):
                        model = SVR(C=i1, cache_size=200, degree=3, epsilon=i4,
                                    gamma=i2, kernel='rbf', max_iter=-1, shrinking=True, tol=i3, verbose=False)
                        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                               base_k=base_k, base_b=base_b, model=model)
                        y_data_test = np.reshape(y_data_test, (-1, 1))
                        y_data_test_corrected = model.predict(y_data_test)
                        mae_name = 'SVR_' + str(i1) + '_' + str(i2) + '_' + str(i3) + '_' + str(i4)
                        x_data_corrected = (y_data_test_corrected - base_b) / base_k
                        self.model_mae[mae_name] = mean_absolute_error(y_data_base, y_data_test_corrected)
                        self.model_svr_mae = self.model_svr_mae.append(
                            {'mae_name': mae_name, 'C': i1, 'gamma': i2, 'tol': i3, 'eps': i4,
                             'mae': self.model_mae[mae_name]}, ignore_index=True)
                        if self.model_mae[mae_name] < self.min_svr_mae:
                            self.min_svr_mae = self.model_mae[mae_name]
                            self.min_svr_c = i1
                            self.min_svr_g = i2
                            self.min_svr_t = i3
                            self.min_svr_e = i4
                        i4 *= 1.5
                    i3 *= 1.5
                i2 *= 1.5
            i1 *= 1.5

    def process(self):
        # training plot
        x_data_training, y_data_training = self.training_plot(training_data_file='/input/noch2-training.csv')
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)

        self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        output_file_path = self._local_dir + '/output/training-2.png'
        plt.savefig(output_file_path)
        plt.show()

        # # # test svr with different parameter values
        self.test_svr(test_data_file='/input/noch2-test.csv', x_data_training=x_data_training,
                      y_data_training=y_data_training, base_k=base_k, base_b=base_b)
        print(self.min_svr_mae)
        print(self.min_svr_c)
        print(self.min_svr_g)
        print(self.min_svr_t)
        print(self.min_svr_e)
        output_file_path = self._local_dir + '/output/' + 'svr_parameter_value-2.csv'
        self.model_svr_mae.to_csv(output_file_path, index=True)
        print(self.model_svr_mae)

        # # test plot
        # for i in range(len(self.models)):
        #     self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        #     self.test_plot(test_data_file='/input/noch2-test.csv', x_data_training=x_data_training,
        #                    y_data_training=y_data_training, model_seq=i, base_k=base_k, base_b=base_b)
        #     output_file_path = self._local_dir + '/output/' + self.models[i].__class__.__name__ + str(i) + '-2.png'
        #     plt.savefig(output_file_path)
        #     plt.show()


if __name__ == '__main__':
    noch = Noch2()
    noch.process()
