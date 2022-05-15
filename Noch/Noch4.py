import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from scipy.optimize import leastsq
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


class Noch4:
    def __init__(self):
        self.num_training_set = 5
        self.training_set_base_seq = 2
        self.training_set_plot_color = ['red', 'gold', 'green', 'black', 'purple', 'blue']
        self.set_size = 14
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.reference_temperature = 30.9

        self.models = [
            # SVR(C=16834.112196028233, cache_size=200, degree=3, epsilon=5.9403192063549134e-05,
            #     gamma=0.05852766346593507, kernel='rbf', max_iter=-1, shrinking=True, tol=0.0077073466292589396,
            #     verbose=False),
            # SVR(C=17.085937499999993, cache_size=200, degree=3, epsilon=0.00020048577321447834,
            #     gamma=0.66666666666666664, kernel='rbf', max_iter=-1, shrinking=True, tol=0.002283658260521167,
            #     verbose=False),
            SVR(C=17.085932701636683804, cache_size=200, degree=3, epsilon=0.00020048577321447834,
                gamma=0.6666666666666664, kernel='rbf', max_iter=-1, shrinking=True, tol=0.002283658260521167,
                verbose=False),
            # SVR(C=50, cache_size=200, degree=3, epsilon=0.1,
            #     gamma=2.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.0000001, verbose=False),
            DecisionTreeRegressor(max_depth=3),
            KNeighborsRegressor(n_neighbors=3)
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
        y_data = df.loc[:, [0, 2]].to_numpy()
        return x_data, y_data

    def base_plot(self, k, b, x_data_base):
        y_data_lf = k * x_data_base + b
        plt.plot(x_data_base, y_data_lf.T, c='darkred')

    def training_plot(self, x_data_training, y_data_training):
        for i in range(self.num_training_set):
            x_data_training_tmp = x_data_training[i * self.set_size:(i + 1) * self.set_size]
            y_data_training_tmp = y_data_training[i * self.set_size:(i + 1) * self.set_size, 1]
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

        y_data_predict = self.models[model_seq].predict(x_data_predict)
        y_data_predict = np.reshape(y_data_predict, (-1, 1))

        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__
        self.model_mae[model_name] = mean_absolute_error(y_data_base_test, y_data_predict)

        x_data_test_size = len(x_data_test)
        print("*****************" + model_name)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i], y_data_test[i, 1], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i], y_data_predict[i], marker='+', color=self.training_set_plot_color[i])
            print(str(x_data_test[i]) + ',' + str(y_data_predict[i]))

        return

    # def mae_parameter_search_on_test(self, x_data_training, y_data_training, x_data_test, y_data_test, base_k, base_b):
    #
    #     # for training: measure to correct
    #     x_data_fit = np.copy(y_data_training)
    #     x_data_fit = np.reshape(x_data_fit, (-1, 1))
    #     # x_data_fit = self.sc.fit_transform(x_data_fit)
    #
    #     y_data_base_training = base_k * x_data_training[:, 1] + base_b
    #     y_data_fit = np.copy(y_data_base_training)
    #     y_data_fit = np.reshape(y_data_fit, (-1, 1))
    #     # y_data_fit = self.sc.fit_transform(y_data_fit)
    #     y_data_fit = np.reshape(y_data_fit, (-1))
    #     self.model_mae['original'] = mean_absolute_error(y_data_base_training, y_data_training)
    #
    #     # for verifying: measure to predict
    #     y_data_base_test = base_k * x_data_test[:, 1] + base_b
    #     x_data_predict = np.copy(y_data_test)
    #     x_data_predict[:, 0] = self.reference_temperature
    #     x_data_predict = np.reshape(x_data_predict, (-1, 1))
    #     # x_data_predict = self.sc.fit_transform(x_data_predict)
    #
    #     c_min = self.step ** -7
    #     c_max = self.step ** 37
    #     print('c:' + str(c_min) + ' - ' + str(c_max))
    #
    #     gamma_min = self.step ** -29
    #     gamma_max = self.step ** 8
    #     print('gamma:' + str(gamma_min) + ' - ' + str(gamma_max))
    #
    #     tol_min = self.step ** -43
    #     tol_max = self.step ** -4
    #     print('tol:' + str(tol_min) + ' - ' + str(tol_max))
    #
    #     epsilon_min = self.step ** -23
    #     epsilon_max = self.step ** 0
    #     print('epsilon:' + str(epsilon_min) + ' - ' + str(epsilon_max))
    #     k = 0
    #     output_file_path = self._local_dir + '/output/' + 'svr_arameter_value-4.csv'
    #     self.min_svr_mae = self.model_mae['original']
    #
    #     C = c_min
    #     while C <= c_max:
    #         gamma = gamma_min
    #         while gamma <= gamma_max:
    #             epsilon = epsilon_min
    #             while epsilon <= epsilon_max:
    #                 tol = tol_min
    #                 while tol <= tol_max:
    #                     model = SVR(C=C, cache_size=200, degree=3, epsilon=epsilon,
    #                                 gamma=gamma, kernel='rbf', max_iter=-1, shrinking=True, tol=tol, verbose=False)
    #                     # fit
    #                     model.fit(X=x_data_fit, y=y_data_fit)
    #
    #                     # predict
    #                     y_data_predict = model.predict(x_data_predict)
    #                     y_data_predict = np.reshape(y_data_predict, (-1, 1))
    #                     # y_data_predict = self.sc.inverse_transform(y_data_predict)
    #
    #                     mae_name = 'SVR_' + str(C) + '_' + str(gamma) + '_' + str(epsilon) + '_' + str(tol)
    #                     self.model_mae[mae_name] = mean_absolute_error(y_data_base_test, y_data_predict)
    #                     self.model_svr_mae = self.model_svr_mae.append(
    #                         {'mae_name': mae_name, 'C': C, 'gamma': gamma, 'eps': epsilon, 'tol': tol,
    #                          'mae': self.model_mae[mae_name]}, ignore_index=True)
    #
    #                     if self.model_mae[mae_name] < self.min_svr_mae:
    #                         self.min_svr_mae = self.model_mae[mae_name]
    #                         self.min_svr_c = C
    #                         self.min_svr_g = gamma
    #                         self.min_svr_e = epsilon
    #                         self.min_svr_t = tol
    #
    #                     if k >= 10000:
    #                         self.model_svr_mae.to_csv(output_file_path, index=True, mode='a', header=True)
    #                         k = 0
    #                         del self.model_svr_mae
    #                         self.model_svr_mae = pd.DataFrame()
    #                         print(time.asctime(time.localtime(time.time())) + ':' + str(self.min_svr_mae))
    #                         print('c=' + str(self.min_svr_c) + ',g=' + str(self.min_svr_g) + ',e=' + str(
    #                             self.min_svr_e) + 't=' + str(self.min_svr_t))
    #
    #                     k += 1
    #
    #                     tol *= self.step
    #                 epsilon *= self.step
    #             gamma *= self.step
    #         C *= self.step

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

    def calcul_original_mae_random(self, x_data_training_mae, y_data_training_mae, loop_num, base_k, base_b):
        original_mae_random = []

        for i in range(loop_num):
            y_data_training_mae_base = base_k * x_data_training_mae[i, :] + base_b
            y_expect = y_data_training_mae_base.tolist()
            y_actual = np.reshape(y_data_training_mae[i], (-1, 2))[:, 1]
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
            y_data_expect = base_k * x_data_training_mae[i, :] + base_b
            x_data_predict = np.copy(y_data_training_mae[i])
            x_data_predict[:, 0] = self.reference_temperature
            x_data_predict = np.reshape(x_data_predict, (-1, 2))
            # x_data_predict = self.sc.fit_transform(x_data_predict)

            y_data_predict = self.models[model_seq].predict(x_data_predict)
            y_data_predict = np.reshape(y_data_predict, (-1, 1))

            mae_tmp = mean_absolute_error(y_data_expect, y_data_predict)
            corrected_mae_random.append(mae_tmp)

        corrected_mae_random_avg = sum(corrected_mae_random) / len(corrected_mae_random)
        return corrected_mae_random, corrected_mae_random_avg

    def set_box_color(self, bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    def box_plot(self, model_seq, x_data_training, y_data_training, base_k, base_b):
        mae_box_model = np.empty((self.set_size, self.num_training_set))
        mae_box_training = np.empty((self.set_size, self.num_training_set))

        plt.figure()

        ticks = np.empty(self.set_size)
        for i in range(self.set_size):
            mae_box_training_tmp = np.empty(self.num_training_set)
            mae_box_model_tmp = np.empty(self.num_training_set)

            for j in range(self.num_training_set):
                x_data_training_tmp = x_data_training[i + j * self.set_size]
                y_data_training_tmp = y_data_training[i + j * self.set_size]
                y_data_base_tmp = base_k * x_data_training_tmp + base_b
                mae_box_training_tmp[j] = mean_absolute_error([y_data_base_tmp], [y_data_training_tmp[1]])

                x_data_predict = np.copy(y_data_training_tmp)
                x_data_predict[0] = self.reference_temperature
                x_data_predict = np.reshape(x_data_predict, (-1, 2))
                y_data_test_tmp = self.models[model_seq].predict(x_data_predict)
                y_data_test_tmp = np.reshape(y_data_test_tmp, (-1, 1))
                mae_box_model_tmp[j] = mean_absolute_error([y_data_base_tmp], y_data_test_tmp)
            mae_box_training[i] = mae_box_training_tmp
            mae_box_model[i] = mae_box_model_tmp
            ticks[i] = str(x_data_training[i])
        mae_box_model_plot = plt.boxplot(mae_box_model.tolist(),
                                         positions=np.array(
                                             np.arange(self.set_size)) * 2.0 + 0.5,
                                         widths=0.6, showfliers=False)
        mae_box_training_plot = plt.boxplot(mae_box_training.tolist(),
                                            positions=np.array(
                                                np.arange(self.set_size)) * 2.0 + 1.5,
                                            widths=0.6, showfliers=False)

        self.set_box_color(mae_box_model_plot, '#D7191C')  # colors are from http://colorbrewer2.org/
        self.set_box_color(mae_box_training_plot, '#2C7BB6')

        plt.xticks(range(0, self.set_size * 2, 2), ticks)
        plt.xlim(-2, self.set_size * 2)

        plt.plot([], c='#D7191C', label='SVR')
        plt.plot([], c='#2C7BB6', label='Without SVR')
        plt.legend()

        plt.tight_layout()

    def process(self):
        data_file_training = '/input/noch2-training.csv'
        x_data_training, y_data_training = self.load_data(data_file=data_file_training)
        data_file_test = '/input/noch2-test.csv'
        x_data_test, y_data_test = self.load_data(data_file=data_file_test)
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)

        y_data_training_base = base_k * x_data_training + base_b
        self.model_mae['original'] = mean_absolute_error(y_data_training_base, y_data_training[:, 1])

        self.training_plot(x_data_training, y_data_training)
        self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        output_file_path = self._local_dir + '/output/training_base.png'
        plt.savefig(output_file_path)
        plt.show()

        training_mae_ind, x_data_training_mae, y_data_training_mae = self.generate_mae_dataset(self.loop_num,
                                                                                               x_data_training,
                                                                                               y_data_training)
        original_mae_random, original_mae_random_avg = self.calcul_original_mae_random(
            x_data_training_mae=x_data_training_mae,
            y_data_training_mae=y_data_training_mae, loop_num=self.loop_num,
            base_k=base_k, base_b=base_b)
        print(original_mae_random)
        print('original_mae_random_avg:' + str(original_mae_random_avg))

        for i in range(len(self.models)):
            self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
            self.fit_plot(model_seq=i, x_data_training=x_data_training, y_data_training=y_data_training,
                          x_data_test=x_data_test, y_data_test=y_data_test,
                          base_k=base_k, base_b=base_b)

            output_file_path = self._local_dir + '/output/' + str(i) + '_' + self.models[i].__class__.__name__ + '.png'
            plt.savefig(output_file_path)
            plt.show()

            corrected_mae_random, corrected_mae_random_avg = self.calcul_corrected_mae_random(
                x_data_training_mae=x_data_training_mae,
                y_data_training_mae=y_data_training_mae, loop_num=self.loop_num,
                base_k=base_k, base_b=base_b, model_seq=i)
            print(corrected_mae_random)
            print(str(i) + ' corrected_mae_random_avg:' + str(corrected_mae_random_avg))

        print(self.model_mae)

        self.box_plot(model_seq=self.favorit_svr_seq, x_data_training=x_data_training, y_data_training=y_data_training,
                      base_k=base_k, base_b=base_b)
        output_file_path = self._local_dir + '/output/box-svr.png'
        plt.savefig(output_file_path)
        plt.show()

        return


if __name__ == '__main__':
    noch = Noch4()
    noch.process()
