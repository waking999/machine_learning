import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


class NotchML:
    def __init__(self):
        self._local_dir = _local_dir = os.path.dirname(__file__)

        self.num_training_set = 11
        self.set_size = 9
        self.training_set_base_seq = self.num_training_set // 2  # the data set is ordered by temperature ascending, we take the middle one

        self.model_mae = {}

        # https://matplotlib.org/stable/gallery/color/named_colors.html
        self.training_set_plot_color = ['red', 'sienna', 'darkorange', 'gold', 'yellow', 'green', 'turquoise',
                                        'cadetblue', 'black', 'purple', 'blue']

        self.models = [
            SVR(C=37876.75244106351, cache_size=200, degree=3, epsilon=0.0010149592268982965,
                gamma=0.9999999999999996, kernel='rbf', max_iter=-1, shrinking=True, tol=0.0022836582605211667,
                verbose=False),
            # SVR(C=19.08374928032402, cache_size=200, degree=3, epsilon=0.05852766346593507,
            #     gamma=2.8834204418832886e-05, kernel='rbf', max_iter=-1, shrinking=True, tol=0.19753086419753085,
            #     verbose=False),
            DecisionTreeRegressor(max_depth=3),
            KNeighborsRegressor(n_neighbors=10)
        ]

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        _df = pd.read_csv(file_path, sep=sep_char, header=header)
        return _df

    def load_data(self, data_file):
        df = self.load_dataset(self._local_dir + data_file, sep_char=',', header=None)

        x_data_plot = df.loc[:, 1].to_numpy()
        y_data_plot = df.loc[:, 2].to_numpy()
        y_data_plot2 = df.loc[:, 3].to_numpy()

        data_fit = df.loc[:, [2, 3]].to_numpy()
        data_tag = df.loc[:, 4].to_numpy()

        return x_data_plot, y_data_plot, y_data_plot2, data_fit, data_tag

    @staticmethod
    def base_plot(x_data_base, y_data_base):
        plt.plot(x_data_base, y_data_base.T, c='darkred')

    def training_plot(self, x_data, y_data):
        for i in range(self.num_training_set):
            x_data_training_tmp = x_data[i * self.set_size:(i + 1) * self.set_size]
            y_data_training_tmp = y_data[i * self.set_size:(i + 1) * self.set_size]
            plt.scatter(x_data_training_tmp, y_data_training_tmp,
                        marker="." if i != self.training_set_base_seq else "x",
                        c=self.training_set_plot_color[i])

    def fit_on_test(self, model_seq, data_fit_training, data_tag_training, x_data_plot_test, y_data_plot_test,
                    data_fit_test, data_tag_test):
        # fit
        self.models[model_seq] = self.models[model_seq].fit(X=data_fit_training, y=data_tag_training)

        # test predict
        data_predict_test = self.models[model_seq].predict(data_fit_test)
        data_predict_test = np.reshape(data_predict_test, (-1, 1))

        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__
        self.model_mae[model_name] = mean_absolute_error(data_tag_test, data_predict_test)

    def fit_plot(self, model_seq, data_fit_training, data_tag_training, x_data_plot_validation, y_data_plot_validation,
                 data_fit_validation, data_tag_validation):
        # fit
        self.models[model_seq] = self.models[model_seq].fit(X=data_fit_training, y=data_tag_training)

        # validation predict
        data_predict_validation = self.models[model_seq].predict(data_fit_validation)
        data_predict_validation = np.reshape(data_predict_validation, (-1, 1))

        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__
        # self.model_mae[model_name] = mean_absolute_error(data_tag_validation, data_predict_validation)

        x_data_plot_validation_size = len(x_data_plot_validation)
        print(model_name)
        for i in range(x_data_plot_validation_size):
            plt.scatter(x_data_plot_validation[i], y_data_plot_validation[i], marker='.',
                        color=self.training_set_plot_color[i])
            plt.scatter(x_data_plot_validation[i], data_predict_validation[i], marker='+',
                        color=self.training_set_plot_color[i])
            # print(str(x_data_plot_validation[i]) + ',' + str(data_predict_validation[i]))

    def process(self):
        data_file_training = './input/notch-training-afterlf.csv'
        x_data_plot_training, y_data_plot_training, y_data_plot_training2, data_fit_training, data_tag_training = self.load_data(
            data_file=data_file_training)
        self.model_mae['original'] = mean_absolute_error(data_tag_training, data_fit_training[:, 0])
        x_data_base = x_data_plot_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        y_data_base = data_tag_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]

        data_file_validation = './input/notch-validation-afterlf.csv'
        x_data_plot_validation, y_data_plot_validation, y_data_plot_validation2, data_fit_validation, data_tag_validation = self.load_data(
            data_file=data_file_validation)

        # pic 1: all temperature and base line
        self.training_plot(x_data_plot_training, y_data_plot_training)
        self.training_plot(x_data_plot_training, y_data_plot_training2)
        self.base_plot(x_data_base, y_data_base)
        plt.show()

        # pic algorithm:
        for i in range(len(self.models)):
            self.base_plot(x_data_base, y_data_base)
            print('algorithm:' + str(i))
            self.fit_plot(model_seq=i, data_fit_training=data_fit_training, data_tag_training=data_tag_training,
                          x_data_plot_validation=x_data_plot_validation, y_data_plot_validation=y_data_plot_validation,
                          data_fit_validation=data_fit_validation, data_tag_validation=data_tag_validation)

            output_file_path = self._local_dir + '/output/' + self.models[i].__class__.__name__ + str(i) + '.png'
            plt.savefig(output_file_path)
            plt.show()

        data_file_test = './input/notch-test-afterlf.csv'
        x_data_plot_test, y_data_plot_test, y_data_plot_test2, data_fit_test, data_tag_test = self.load_data(
            data_file=data_file_test)

        # performance algorithm:
        for i in range(len(self.models)):
            self.base_plot(x_data_base, y_data_base)
            print('algorithm:' + str(i))
            self.fit_on_test(model_seq=i, data_fit_training=data_fit_training, data_tag_training=data_tag_training,
                             x_data_plot_test=x_data_plot_test,
                             y_data_plot_test=y_data_plot_test,
                             data_fit_test=data_fit_test, data_tag_test=data_tag_test)

        # show performance of each algorithm
        print(self.model_mae)


if __name__ == '__main__':
    notch = NotchML()
    notch.process()
