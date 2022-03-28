# import math
import os
import time
import random

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
        self.training_set_plot_color = ['red', 'gold', 'green', 'black', 'purple','blue']
        self.set_size = 13
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.models = [

            # SVR(C=50, cache_size=200, degree=3, epsilon=0.1,
            #     gamma=2.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # SVR(C=0.005, cache_size=200, degree=3, epsilon=0.00001,
            #     gamma=0.0001, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            SVR(C=11222.7414640188, cache_size=200, degree=3, epsilon=0.00152243884034744,
                gamma=5.06249999999999, kernel='rbf', max_iter=-1, shrinking=True, tol=0.000200485773214478,
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
        # measured frequency
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

        model_name = model.__class__.__name__ + str(model_seq)
        self.model_mae[model_name] = mean_absolute_error(y_data_base, y_data_test_corrected)

        x_data_test_size = len(x_data_test)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_test_corrected[i], marker='+', color=self.training_set_plot_color[i])

            stress_predicted = (y_data_test_corrected[i] - base_b) / base_k
            # print(str(x_data_test[i, 1]) + ',' + str(y_data_test_measured[i]) + ',' + str(y_data_base[i]) + ',' + str(
            #     y_data_test_corrected[i]) + ',' + str(stress_predicted))

    def mae_parameter_search(self, test_data_file, x_data_training, y_data_training, base_k, base_b):
        # df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        # x_data_test = df.loc[:, 0:1].to_numpy()
        # y_data_test = df.loc[:, 2].to_numpy()
        y_data_base = base_k * x_data_training[:, 1] + base_b
        step = 1.5



        parameters = [{
            'C': np.logspace(base=step, start=7, stop=37, num=(37 - 7 + 1), endpoint=True),
            'gamma': np.logspace(base=step, start=-15, stop=8, num=(8 + 15 + 1), endpoint=True),
            'tol': np.logspace(base=step, start=-42, stop=-4, num=(42 - 4 + 1), endpoint=True),
            'epsilon': np.logspace(base=step, start=-17, stop=0, num=(0 + 17 + 1), endpoint=True)
        }]
        print(parameters)

        clf = GridSearchCV(
            SVR(kernel='rbf', shrinking=True, degree=3, cache_size=200, max_iter=-1),
            param_grid=parameters,
            scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1
        )

        # # corrected frequency
        # y_data_corrected = base_k * x_data_training[:, 1] + base_b
        # # measured frequencey
        # y_data_measured = np.reshape(y_data_training, (-1, 1))
        #
        # clf.fit(X=y_data_measured, y=y_data_corrected)

        # print('clf.best_params_', clf.best_params_)

        c_min = step ** -14
        c_max = step ** 37
        print('c:' + str(c_min) + ' - ' + str(c_max))

        gamma_min = step ** -36
        gamma_max = step ** 8
        print('gamma:' + str(gamma_min) + ' - ' + str(gamma_max))

        tol_min = step ** -50
        tol_max = step ** -4
        print('tol:' + str(tol_min) + ' - ' + str(tol_max))

        epsilon_min = step ** -30
        epsilon_max = step ** 0
        print('epsilon:' + str(epsilon_min) + ' - ' + str(epsilon_max))

        y_data_training = np.reshape(y_data_training, (-1, 1))

        i1 = c_min
        self.min_svr_mae = 1

        while i1 <= c_max:
            i2 = gamma_min
            while i2 <= gamma_max:
                i3 = tol_min
                while i3 <= tol_max:
                    i4 = epsilon_min
                    while i4 <= epsilon_max:
                        model = SVR(C=i1, cache_size=200, degree=3, epsilon=i4,
                                    gamma=i2, kernel='rbf', max_iter=-1, shrinking=True, tol=i3, verbose=False)
                        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                               base_k=base_k, base_b=base_b, model=model)

                        y_data_training_corrected = model.predict(y_data_training)
                        mae_name = 'SVR_' + str(i1) + '_' + str(i2) + '_' + str(i3) + '_' + str(i4)
                        # x_data_corrected = (y_data_training_corrected - base_b) / base_k
                        self.model_mae[mae_name] = mean_absolute_error(y_data_base, y_data_training_corrected)
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

    def calcul_original_mae_random(self, x_data_training_mae, y_data_training_mae, loop_num, base_k, base_b):
        original_mae_random = []

        for i in range(loop_num):
            y_data_training_mae_base = base_k * x_data_training_mae[i, :, 1] + base_b
            y_expect = y_data_training_mae_base.tolist()
            y_actual = y_data_training_mae[i]
            # print(y_expect)
            # print(y_actual)
            mae_tmp = mean_absolute_error(y_expect, y_actual)
            original_mae_random.append(mae_tmp)

        original_mae_random_avg = sum(original_mae_random) / len(original_mae_random)
        return original_mae_random, original_mae_random_avg

    def calcul_corrected_mae_random(self, x_data_training_mae, y_data_training_mae, loop_num, base_k, base_b,
                                    model_seq):
        corrected_mae_random = []
        for i in range(loop_num):
            y_data_training_mae_base = base_k * x_data_training_mae[i, :, 1] + base_b
            y_expect = y_data_training_mae_base.tolist()
            y_actual = self.models[model_seq].predict(np.reshape(y_data_training_mae[i], (-1, 1)))
            # print(y_expect)
            # print(y_data_training_mae[i])
            # print(y_actual)
            mae_tmp = mean_absolute_error(y_expect, y_actual)
            corrected_mae_random.append(mae_tmp)


        corrected_mae_random_avg = sum(corrected_mae_random) / len(corrected_mae_random)
        return corrected_mae_random, corrected_mae_random_avg

    def generate_mae_dataset(self, loop_num, x_data_training, y_data_training):
        traing_mae_ind = []
        x_data_training_mae = []
        y_data_training_mae = []
        for i in range(loop_num):
            # generate 5 test data (because 5 temperature series)
            traing_mae_ind_tmp = []
            x_data_training_mae_tmp = []
            y_data_training_mae_tmp = []
            for j in range(self.num_training_set):
                range_start = j * self.set_size
                range_end = (j + 1) * self.set_size
                rand_ind = random.randrange(range_start, range_end)
                traing_mae_ind_tmp.append(rand_ind)
                x_data_training_mae_tmp.append(x_data_training[rand_ind])
                y_data_training_mae_tmp.append(y_data_training[rand_ind])
            traing_mae_ind.append(traing_mae_ind_tmp)
            x_data_training_mae.append(x_data_training_mae_tmp)
            y_data_training_mae.append(y_data_training_mae_tmp)

        return traing_mae_ind, np.array(x_data_training_mae), y_data_training_mae



    def process(self):
        # training plot
        x_data_training, y_data_training = self.training_plot(training_data_file='/input/noch2-training.csv')
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)

        self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        output_file_path = self._local_dir + '/output/training-2.png'
        plt.savefig(output_file_path)
        plt.show()

        # # # test svr with different parameter values
        t1 = time.time()
        self.mae_parameter_search(test_data_file='/input/noch2-test.csv', x_data_training=x_data_training,
                                  y_data_training=y_data_training, base_k=base_k, base_b=base_b)
        t2 = time.time()
        print('parameter value spends ' + str((t2 - t1) / 60) + 's')
        print(self.min_svr_mae)
        print(self.min_svr_c)
        print(self.min_svr_g)
        print(self.min_svr_t)
        print(self.min_svr_e)
        output_file_path = self._local_dir + '/output/' + 'svr_parameter_value-2.csv'
        self.model_svr_mae.to_csv(output_file_path, index=True)
        print(self.model_svr_mae)

        # calculate random sample mae
        # traing_mae_ind, x_data_training_mae, y_data_training_mae = self.generate_mae_dataset(self.loop_num,
        #                                                                                      x_data_training,
        #                                                                                      y_data_training)
        # print(traing_mae_ind)
        # print('----')
        # print(x_data_training_mae)
        # print('----')
        # print(y_data_training_mae)
        # print('----')
        # original_mae_random, original_mae_random_avg = self.calcul_original_mae_random(
        #     x_data_training_mae=x_data_training_mae,
        #     y_data_training_mae=y_data_training_mae, loop_num=self.loop_num,
        #     base_k=base_k, base_b=base_b)
        # print(original_mae_random)
        # print('original_mae_random_avg:' + str(original_mae_random_avg))

        # # test plot
        # for i in range(len(self.models)):
        #     self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        #     self.test_plot(test_data_file='/input/noch2-test.csv', x_data_training=x_data_training,
        #                    y_data_training=y_data_training, model_seq=i, base_k=base_k, base_b=base_b)
        #     output_file_path = self._local_dir + '/output/' + self.models[i].__class__.__name__ + str(i) + '-2.png'
        #     plt.savefig(output_file_path)
        #     plt.show()
        #
        #     corrected_mae_random, corrected_mae_random_avg = self.calcul_corrected_mae_random(
        #         x_data_training_mae=x_data_training_mae,
        #         y_data_training_mae=y_data_training_mae, loop_num=self.loop_num,
        #         base_k=base_k, base_b=base_b, model_seq=i)
        #     print(corrected_mae_random)
        #     print(str(i) + ' corrected_mae_random_avg:' + str(original_mae_random_avg))
        #
        #
        #
        #
        # print(self.model_mae)


if __name__ == '__main__':
    noch = Noch2()
    noch.process()
