import os

import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
# import xgboost as xgb
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class Noch1:
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
            SVR(C=500, cache_size=200, degree=3, epsilon=0.0001,
                gamma=1.00E-05, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
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

    def training_test_plot_box_mae(self, x_data_training, y_data_training, base_k, base_b, model):
        mae_box_test = np.empty((self.set_size, self.num_training_set))
        mae_box_training = np.empty((self.set_size, self.num_training_set))

        plt.figure()

        ticks = np.empty(self.set_size)
        for i in range(self.set_size):
            mae_box_training_tmp = np.empty(self.num_training_set)
            mae_box_test_tmp = np.empty(self.num_training_set)
            for j in range(self.num_training_set):
                x_data_training_tmp = x_data_training[i + j * self.set_size, 1]
                y_data_training_tmp = y_data_training[i + j * self.set_size]
                y_data_base_tmp = base_k * x_data_training_tmp + base_b
                mae_box_training_tmp[j] = mean_absolute_error([y_data_base_tmp], [y_data_training_tmp])
                y_data_test_tmp = model.predict(np.array([[self.temperature_base, x_data_training_tmp]]))
                mae_box_test_tmp[j] = mean_absolute_error([y_data_base_tmp], [y_data_test_tmp])
            mae_box_training[i] = mae_box_training_tmp
            mae_box_test[i] = mae_box_test_tmp
            ticks[i] = str(x_data_training[i, 1])

        mae_box_test_plot = plt.boxplot(mae_box_test.tolist(),
                                        positions=np.array(
                                            np.arange(self.set_size)) * 2.0 + 0.5,
                                        widths=0.6, showfliers=False)
        mae_box_training_plot = plt.boxplot(mae_box_training.tolist(),
                                            positions=np.array(
                                                np.arange(self.set_size)) * 2.0 + 1.5,
                                            widths=0.6, showfliers=False)
        self.set_box_color(mae_box_test_plot, '#D7191C')  # colors are from http://colorbrewer2.org/
        self.set_box_color(mae_box_training_plot, '#2C7BB6')

        plt.xticks(range(0, self.set_size * 2, 2), ticks)
        plt.xlim(-2, self.set_size * 2)

        plt.plot([], c='#D7191C', label='SVR')
        plt.plot([], c='#2C7BB6', label='Without SVR')
        plt.legend()

        plt.tight_layout()

    def training_plot_3d(self, x_data_training, y_data_training):
        x = np.unique(x_data_training[:, 0])
        y = np.unique(x_data_training[:, 1])
        Y, X = np.meshgrid(y, x)
        Z = y_data_training.reshape((len(x), len(y)))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='rainbow')
        ax.scatter3D(X, Y, Z)

    def linear_fitting(self, x_data, y_data):
        p0 = np.array([100, 20])
        x_data = x_data.astype('float')
        y_data = y_data.astype('float')
        ret = leastsq(self.err, p0, args=(x_data, y_data))
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

    def model_fit(self, x_data_training, y_data_training, fix_temperature, model):
        x_data_training_notemp = x_data_training.copy()
        x_data_training_notemp[:, 0] = fix_temperature
        model.fit(X=x_data_training_notemp, y=y_data_training)
        return model

    def test_plot(self, test_data_file, x_data_training, y_data_training, model_seq, base_k, base_b):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()

        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                               fix_temperature=self.temperature_base,
                               model=self.models[model_seq])
        y_data_model_temp = model.predict(x_data_test)

        y_data_base = base_k * x_data_test + base_b
        self.model_mae[model.__class__.__name__ + str(model_seq)] = mean_absolute_error(y_data_base[:, 1],
                                                                                        y_data_model_temp)

        x_data_test_size = len(x_data_test)
        for i in range(x_data_test_size):
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_model_temp[i], marker='+', color=self.training_set_plot_color[i])

    def test_lf_plot(self, test_data_file, x_data_training, y_data_training, base_k, base_b):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_test = df.loc[:, 2].to_numpy()
        y_data_base = base_k * x_data_test + base_b

        self.model_mae["Origin"] = mean_absolute_error(y_data_base[:, 1], y_data_test)

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

        sec_layer_k, sec_layer_b = self.linear_fitting(x_data_test_model[1:, 1], y_data_test_model[1:])

        y_data_sec_layer = np.empty(x_data_test_size)
        for i in range(x_data_test_size):
            y_data_test_lf_predict = sec_layer_k * x_data_test[i, 1] + sec_layer_b
            plt.scatter(x_data_test[i, 1], y_data_test[i], marker='.', color=self.training_set_plot_color[i])
            plt.scatter(x_data_test[i, 1], y_data_test_lf_predict, marker='+', color=self.training_set_plot_color[i])
            y_data_sec_layer[i] = y_data_test_lf_predict

        self.model_mae["SecondLayerLinearFitting"] = mean_absolute_error(y_data_base[:, 1], y_data_sec_layer)

    def test_svr(self, test_data_file, x_data_training, y_data_training, base_k, base_b):
        df = self.load_dataset(self._local_dir + test_data_file, sep_char=',', header=None)
        x_data_test = df.loc[:, 0:1].to_numpy()
        y_data_base = base_k * x_data_test + base_b

        i1 = 0.005
        self.min_svr_mae = 1
        while i1 <= 500:
            i2 = 0.00001
            while i2 <= 0.1:
                i3 = 0.0000001
                while i3 <= 0.1:
                    i4 = 0.00001
                    while i4 <= 0.01:
                        model = SVR(C=i1, cache_size=200, degree=3, epsilon=i4,
                                    gamma=i2, kernel='rbf', max_iter=-1, shrinking=True, tol=i3, verbose=False)
                        model = self.model_fit(x_data_training=x_data_training, y_data_training=y_data_training,
                                               fix_temperature=self.temperature_base,
                                               model=model)
                        y_data_model_temp = model.predict(x_data_test)
                        mae_name = 'SVR_' + str(i1) + '_' + str(i2) + '_' + str(i3) + '_' + str(i4)
                        self.model_mae[mae_name] = mean_absolute_error(y_data_base[:, 1], y_data_model_temp)
                        self.model_svr_mae = self.model_svr_mae.append(
                            {'mae_name': mae_name, 'C': i1, 'gamma': i2, 'tol': i3, 'eps': i4,
                             'mae': self.model_mae[mae_name]}, ignore_index=True)
                        if self.model_mae[mae_name] < self.min_svr_mae:
                            self.min_svr_mae = self.model_mae[mae_name]
                            self.min_svr_c = i1
                            self.min_svr_g = i2
                            self.min_svr_t = i3
                            self.min_svr_e = i4
                        i4 *= 10
                    i3 *= 10
                i2 *= 10
            i1 *= 10

    def process(self):
        # training plot
        x_data_training, y_data_training = self.training_plot(training_data_file='/input/noch1-training.csv')
        base_k, base_b, x_data_base = self.calculate_base_kb(x_data_training, y_data_training)

        self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        output_file_path = self._local_dir + '/output/training.png'
        plt.savefig(output_file_path)
        plt.show()

        self.training_plot_3d(x_data_training, y_data_training)
        output_file_path = self._local_dir + '/output/3d.png'
        plt.savefig(output_file_path)
        plt.show()

        # test plot
        for i in range(len(self.models)):
            self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)

            self.test_plot(test_data_file='/input/noch1-test.csv', x_data_training=x_data_training,
                           y_data_training=y_data_training, model_seq=i, base_k=base_k, base_b=base_b)

            output_file_path = self._local_dir + '/output/' + self.models[i].__class__.__name__ +str(i)+ '.png'
            plt.savefig(output_file_path)
            plt.show()

        # test lf plot
        self.base_plot(k=base_k, b=base_b, x_data_base=x_data_base)
        self.test_lf_plot(test_data_file='/input/noch1-test.csv', x_data_training=x_data_training,
                          y_data_training=y_data_training, base_k=base_k, base_b=base_b)

        output_file_path = self._local_dir + '/output/test_lf.png'
        plt.savefig(output_file_path)
        plt.show()

        # # test svr with different parameter values
        # self.test_svr(test_data_file='/input/noch1-test.csv', x_data_training=x_data_training,
        #               y_data_training=y_data_training, base_k=base_k, base_b=base_b)
        # print(self.min_svr_mae)
        # print(self.min_svr_c)
        # print(self.min_svr_g)
        # print(self.min_svr_t)
        # print(self.min_svr_e)
        # output_file_path = self._local_dir + '/output/' + 'svr_parameter_value.csv'
        # self.model_svr_mae.to_csv(output_file_path, index=True)
        # print(self.model_svr_mae)

        self.training_test_plot_box_mae(x_data_training, y_data_training, base_k, base_b,
                                        model=self.models[self.favorit_svr_seq])
        output_file_path = self._local_dir + '/output/training-test-box.png'
        plt.savefig(output_file_path)
        plt.show()

        print(self.model_mae)


if __name__ == '__main__':
    noch = Noch1()
    noch.process()