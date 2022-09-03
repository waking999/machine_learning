import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from scipy.optimize import leastsq
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

"""
use fm1, fm2 as x, search svr parameter value
"""


class Notch10:
    def __init__(self):
        self.num_training_set = 5
        self.training_set_base_seq = 2
        self.training_set_plot_color = ['red', 'gold', 'green', 'black', 'purple', 'blue']
        self.set_size = 14
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.reference_temperature = 30.9

        self.models = [
            SVR(C=17.085937499999993, cache_size=200, degree=3, epsilon=0.00020048577321447834,
                gamma=0.6666666666666664, kernel='rbf', max_iter=-1, shrinking=True, tol=0.002283658260521167,
                verbose=False),
            DecisionTreeRegressor(max_depth=3),
            KNeighborsRegressor(n_neighbors=10)
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
        print('R2:' + str(r2_score(y_data, y_data_lr)))
        print('k,b=' + str(ret))
        return ret[0]

    def calculate_base_kb(self, x_data_training, y_data_training):
        x_data_base = x_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        y_data_base = y_data_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size, 1]
        k, b = self.linear_fitting(x_data_base, y_data_base)
        return k, b, x_data_base

    def load_data(self, data_file):
        df = self.load_dataset(self._local_dir + data_file, sep_char=',', header=None)
        x_data = df.loc[:, 1].to_numpy()
        y_data = df.loc[:, [2, 3]].to_numpy()
        return x_data, y_data

    def base_plot(self, k, b, x_data_base):
        y_data_lf = k * x_data_base + b
        plt.plot(x_data_base, y_data_lf.T, c='darkred')

    def training_plot(self, x_data_training, y_data_training):
        for i in range(self.num_training_set):
            x_data_training_tmp = x_data_training[i * self.set_size:(i + 1) * self.set_size, 1]
            y_data_training_tmp = y_data_training[i * self.set_size:(i + 1) * self.set_size]
            plt.scatter(x_data_training_tmp, y_data_training_tmp,
                        marker="." if i != self.training_set_base_seq else "x",
                        c=self.training_set_plot_color[i])

    def fit(self, x_data_training, y_data_training, model_seq, base_k, base_b):

        # for training: measure to correct
        x_data_fit = np.copy(y_data_training)
        x_data_fit = np.reshape(x_data_fit, (-1, 2))

        y_data_base_training = base_k * x_data_training + base_b
        y_data_fit = np.copy(y_data_base_training)

        self.models[model_seq] = self.models[model_seq].fit(X=x_data_fit, y=y_data_fit)

    def fit_plot(self, model_seq, x_data_training, y_data_training, x_data_test, y_data_test, base_k, base_b):
        # fit
        self.fit(x_data_training, y_data_training, model_seq, base_k, base_b)

        # for verifying: measure to predict
        y_data_base_test = base_k * x_data_test + base_b
        x_data_predict = np.copy(y_data_test)
        x_data_predict[:, 0] = self.reference_temperature
        x_data_predict = np.reshape(x_data_predict, (-1, 2))

        # predict
        y_data_predict = self.models[model_seq].predict(x_data_predict)
        y_data_predict = np.reshape(y_data_predict, (-1, 1))

        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__
        self.model_mae[model_name] = mean_absolute_error(y_data_base_test, y_data_predict)

        x_data_test_size = len(x_data_test)
        print(model_name)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i], y_data_test[i, 1], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i], y_data_predict[i], marker='+', color=self.training_set_plot_color[i])
            print(str(x_data_test[i]) + ',' + str(y_data_predict[i, 0]))

    def mae_verification_on_training(self, x_data_training, y_data_training, base_k, base_b, model_seq):
        model = self.models[model_seq]
        x_data_fit = np.copy(y_data_training)
        x_data_fit = np.reshape(x_data_fit, (-1, 2))
        y_data_base_training = base_k * x_data_training + base_b
        y_data_fit = np.copy(y_data_base_training)

        # fit
        model.fit(X=x_data_fit, y=y_data_fit)

        x_data_predict = np.copy(y_data_training)
        x_data_predict[:, 0] = self.reference_temperature
        x_data_predict = np.reshape(x_data_predict, (-1, 2))
        # predict
        y_data_predict = model.predict(x_data_predict)
        y_data_predict = np.reshape(y_data_predict, (-1, 1))

        y_data_base_test = base_k * x_data_training + base_b
        whole_mae = mean_absolute_error(y_data_base_test, y_data_predict)
        print('Model {}: whole_mae={}'.format(model_seq, whole_mae))

    def mae_parameter_search_on_training(self, x_data_training, y_data_training, x_data_test, y_data_test, base_k,
                                         base_b):

        # for training: measure to correct
        x_data_fit = np.copy(y_data_training)
        x_data_fit = np.reshape(x_data_fit, (-1, 2))

        y_data_base_training = base_k * x_data_training + base_b
        y_data_fit = np.copy(y_data_base_training)
        # y_data_fit = np.reshape(y_data_fit, (-1, 1))

        # for verifying: measure to predict
        y_data_base_test = base_k * x_data_training + base_b
        x_data_predict = np.copy(y_data_training)
        # y_data_base_test = base_k * x_data_test + base_b
        # x_data_predict = np.copy(y_data_test)
        ## x_data_predict[:, 0] = self.reference_temperature
        x_data_predict = np.reshape(x_data_predict, (-1, 2))

        # parameters = [{
        #     'C': np.logspace(base=self.step, start=-7, stop=37, num=(37 + 7 + 1), endpoint=True),
        #     'gamma': np.logspace(base=self.step, start=-29, stop=8, num=(8 + 15 + 1), endpoint=True),
        #     'tol': np.logspace(base=self.step, start=-42, stop=-4, num=(42 - 4 + 1), endpoint=True),
        #     'epsilon': np.logspace(base=self.step, start=-17, stop=0, num=(0 + 17 + 1), endpoint=True)
        # }]
        # print(parameters)

        # clf = GridSearchCV(
        #     SVR(kernel='rbf', shrinking=True, degree=3, cache_size=200, max_iter=-1),
        #     param_grid=parameters,
        #     scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1
        # )
        #
        # clf.fit(X=x_data_predict, y=y_data_base_test)
        # print('clf.best_params_', clf.best_params_)

        c_min = self.step ** -47
        c_max = self.step ** 27
        print('c:' + str(c_min) + ' - ' + str(c_max))

        # c_log = np.logspace(base=self.step, start=-7, stop=7, num=(7 + 7 + 1), endpoint=True)
        # print(c_log)

        gamma_min = self.step ** -39
        gamma_max = self.step ** 28
        print('gamma:' + str(gamma_min) + ' - ' + str(gamma_max))

        epsilon_min = self.step ** -38
        epsilon_max = self.step ** 20
        print('epsilon:' + str(epsilon_min) + ' - ' + str(epsilon_max))

        tol_min = self.step ** -34
        tol_max = self.step ** 16
        print('tol:' + str(tol_min) + ' - ' + str(tol_max))

        k = 0
        output_file_path = self._local_dir + '/output/' + 'svr_parameter_value-10b.csv'
        self.min_svr_mae = self.model_mae['original']

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
                        # fit
                        model.fit(X=x_data_fit, y=y_data_fit)

                        # predict
                        y_data_predict = model.predict(x_data_predict)
                        y_data_predict = np.reshape(y_data_predict, (-1, 1))

                        mae_name = 'SVR_' + str(C) + '_' + str(gamma) + '_' + str(epsilon) + '_' + str(tol)
                        self.model_mae[mae_name] = mean_absolute_error(y_data_base_test, y_data_predict)
                        # self.model_svr_mae = self.model_svr_mae.append(
                        #     {'mae_name': mae_name, 'C': C, 'gamma': gamma, 'eps': epsilon, 'tol': tol,
                        #      'mae': self.model_mae[mae_name]}, ignore_index=True)

                        if self.model_mae[mae_name] < self.min_svr_mae:
                            self.min_svr_mae = self.model_mae[mae_name]
                            self.min_svr_c = C
                            self.min_svr_g = gamma
                            self.min_svr_e = epsilon
                            self.min_svr_t = tol
                            self.model_svr_mae = self.model_svr_mae.append(
                                {'mae_name': mae_name, 'C': C, 'gamma': gamma, 'eps': epsilon, 'tol': tol,
                                 'mae': self.model_mae[mae_name]}, ignore_index=True)

                        if k >= 10000:
                            if self.model_svr_mae is not None:
                                self.model_svr_mae.to_csv(output_file_path, index=True, mode='a', header=True)
                                del self.model_svr_mae
                                self.model_svr_mae = pd.DataFrame()

                            k = 0

                            print(time.asctime(time.localtime(time.time())) + ':' + str(self.min_svr_mae))
                            print('c=' + str(self.min_svr_c) + ',g=' + str(self.min_svr_g) + ',e=' + str(
                                self.min_svr_e) + ',t=' + str(self.min_svr_t))

                        k += 1

                        tol *= self.step
                    epsilon *= self.step
                gamma *= self.step
            C *= self.step

    def process(self):
        data_file_training = '/input/notch3-training.csv'
        x_data_training, y_data_training = self.load_data(data_file=data_file_training)
        data_file_test = '/input/notch3-test.csv'
        x_data_test, y_data_test = self.load_data(data_file=data_file_test)
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)

        y_data_training_base = base_k * x_data_training[:] + base_b
        self.model_mae['original'] = mean_absolute_error(y_data_training_base, y_data_training[:, 1])

        t1 = time.time()
        self.mae_parameter_search_on_training(x_data_training=x_data_training, y_data_training=y_data_training,
                                              x_data_test=x_data_test, y_data_test=y_data_test, base_k=base_k,
                                              base_b=base_b)
        t2 = time.time()
        print('mae parameter search spends:' + str((t2 - t1) / 3600) + ' hrs')
        print('parameter value spends ' + str((t2 - t1)) + 's')
        print(self.min_svr_mae)
        print(self.min_svr_c)
        print(self.min_svr_g)
        print(self.min_svr_t)
        print(self.min_svr_e)
        print(self.model_mae)

        # for i in range(len(self.models)):
        #     self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        #     print('algorithm:' + str(i))
        #     self.fit_plot(model_seq=i, x_data_training=x_data_training,
        #                   y_data_training=y_data_training, x_data_test=x_data_test,
        #                   y_data_test=y_data_test, base_k=base_k, base_b=base_b)
        #
        #     output_file_path = self._local_dir + '/output/' + self.models[i].__class__.__name__ + str(i) + '.png'
        #     plt.savefig(output_file_path)
        #     plt.show()
        #
        #     self.mae_verification_on_training(x_data_training=x_data_training,
        #                                       y_data_training=y_data_training, base_k=base_k, base_b=base_b,model_seq=i)
        #
        # print(self.model_mae)


if __name__ == '__main__':
    noch = Notch10()
    noch.process()
