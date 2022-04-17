import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import leastsq
from sklearn.metrics import r2_score


class Noch3:
    def __init__(self):
        self.num_training_set = 5
        self.training_set_base_seq = 2
        self.temperature_base = 30.9
        self.training_set_plot_color = ['red', 'gold', 'green', 'black', 'purple', 'blue']
        self.set_size = 13
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.models = [

            # SVR(C=50, cache_size=200, degree=3, epsilon=0.1,
            #     gamma=2.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # SVR(C=0.005, cache_size=200, degree=3, epsilon=0.00001,
            #     gamma=0.0001, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            SVR(C=11222.741464018818, cache_size=200, degree=3, epsilon=0.001522438840347445,
                gamma=25.62890624999999, kernel='rbf', max_iter=-1, shrinking=True, tol=5.29310884100836e-09,
                verbose=False),
            # DecisionTreeRegressor(max_depth=12),
            # KNeighborsRegressor(n_neighbors=3)
            # # xgb.XGBRegressor(max_depth=127, learning_rate=0.001, n_estimators=1000,
            # #                  objective='reg:tweedie', n_jobs=-1, booster='gbtree'),
            # LinearRegression(),
            # RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
        ]
        self.model_mae = {}
        self.favorit_svr_seq = 0

        self.step = 1.5
        self.model_svr_mae = pd.DataFrame()
        self.min_svr_mae = None
        self.min_svr_c = None
        self.min_svr_g = None
        self.min_svr_t = None
        self.min_svr_e = None

        self.loop_num = 10

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        _df = pd.read_csv(file_path, sep=sep_char, header=header)
        return _df

    def err(self, p, x, y):
        return p[0] * x + p[1] - y

    def linear_fitting(self, x_data, y_data):
        p0 = np.array([100, 20])
        x_data = x_data.astype('float')
        y_data = y_data.astype('float')
        ret = leastsq(self.err, p0, args=(x_data, y_data))

        y_data_lr = ret[0][0] * x_data + ret[0][1]
        print(r2_score(y_data, y_data_lr))

        print(ret)
        return ret[0]

    def calculate_base_kb(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size, 1]
        y_data_base = y_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        k, b = self.linear_fitting(x_data_base, y_data_base)
        return k, b, x_data_base

    def load_data(self, data_file):
        df = self.load_dataset(self._local_dir + data_file, sep_char=',', header=None)
        x_data = df.loc[:, 0:1].to_numpy()
        y_data = df.loc[:, 2].to_numpy()
        return x_data, y_data

    def base_plot(self, k, b, x_data_base):
        y_data_lf = k * x_data_base + b
        plt.plot(x_data_base, y_data_lf.T, c='darkred')

    def mae_parameter_search(self, x_data_training, y_data_training, base_k, base_b):
        # calculate original mae
        y_data_base = base_k * x_data_training[:, 1] + base_b
        self.model_mae['original'] = mean_absolute_error(y_data_base, y_data_training)

        sc_x = StandardScaler()
        sc_y = StandardScaler()

        # for training: measure to correct
        x_data_fit = np.copy(y_data_training)
        x_data_fit = np.reshape(x_data_fit, (-1, 1))
        x_data_fit = sc_x.fit_transform(x_data_fit)

        y_data_fit = np.copy(y_data_base)
        y_data_fit = np.reshape(y_data_fit, (-1, 1))
        y_data_fit = sc_y.fit_transform(y_data_fit)
        y_data_fit = np.reshape(y_data_fit, (-1))

        # for verifying: measure to predict
        x_data_predict = np.copy(y_data_training)
        x_data_predict = np.reshape(x_data_predict, (-1, 1))
        x_data_predict = sc_x.fit_transform(x_data_predict)

        c_min = self.step ** -7
        c_max = self.step ** 37
        # c_max = self.step ** -6
        print('c:' + str(c_min) + ' - ' + str(c_max))

        gamma_min = self.step ** -29
        gamma_max = self.step ** 8
        # gamma_max = self.step ** -28
        print('gamma:' + str(gamma_min) + ' - ' + str(gamma_max))

        tol_min = self.step ** -43
        tol_max = self.step ** -4
        # tol_max = self.step ** -42
        print('tol:' + str(tol_min) + ' - ' + str(tol_max))

        epsilon_min = self.step ** -23
        epsilon_max = self.step ** 0
        # epsilon_max = self.step ** -22
        print('epsilon:' + str(epsilon_min) + ' - ' + str(epsilon_max))

        C = c_min
        while C <= c_max:
            gamma = gamma_min
            while gamma <= gamma_max:
                epsilon = epsilon_min
                while epsilon <= epsilon_max:
                    tol = tol_min
                    while tol <= tol_max:
                        model = SVR(C=C, cache_size=200, degree=3, epsilon=epsilon,
                                    gamma=gamma, kernel='rbf', max_iter=-1, shrinking=True, tol=tol, verbose=False)

                        model.fit(X=x_data_fit, y=y_data_fit)

                        y_data_predict = model.predict(x_data_predict)
                        y_data_predict = np.reshape(y_data_predict, (-1, 1))
                        mae_name = 'SVR_' + str(C) + '_' + str(gamma) + '_' + str(epsilon) + '_' + str(tol)

                        y_data_predict = sc_y.inverse_transform(y_data_predict)

                        self.model_mae[mae_name] = mean_absolute_error(y_data_base, y_data_predict)

                        self.model_svr_mae = self.model_svr_mae.append(
                            {'mae_name': mae_name, 'C': C, 'gamma': gamma, 'eps': epsilon, 'tol': tol,
                             'mae': self.model_mae[mae_name]}, ignore_index=True)
                        if self.model_mae[mae_name] < self.model_mae['original']:
                            self.min_svr_mae = self.model_mae[mae_name]
                            self.min_svr_c = C
                            self.min_svr_g = gamma
                            self.min_svr_e = epsilon
                            self.min_svr_t = tol

                        tol *= self.step
                    epsilon *= self.step
                gamma *= self.step
            C *= self.step

    def process(self):
        data_file_training = '/input/noch2-training.csv'
        x_data_training, y_data_training = self.load_data(data_file=data_file_training)
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)
        # self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        # plt.show()

        t1 = time.time()
        self.mae_parameter_search(x_data_training, y_data_training, base_k, base_b)
        t2 = time.time()
        print('parameter value spends ' + str((t2 - t1) / 3600) + ' hours')
        print(self.min_svr_mae)
        print(self.min_svr_c)
        print(self.min_svr_g)
        print(self.min_svr_t)
        print(self.min_svr_e)
        output_file_path = self._local_dir + '/output/' + 'svr_parameter_value-3.csv'
        self.model_svr_mae.to_csv(output_file_path, index=True)
        print(self.model_svr_mae)
        print(self.model_mae)


if __name__ == '__main__':
    noch = Noch3()
    noch.process()
