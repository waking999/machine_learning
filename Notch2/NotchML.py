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
            SVR(C=129.74633789062494, cache_size=200, degree=3, epsilon=0.0010149592268982965,
                gamma=11.390624999999996, kernel='rbf', max_iter=-1, shrinking=True, tol=4.019454526140724e-08,
                verbose=False),
            # DecisionTreeRegressor(max_depth=3),
            # KNeighborsRegressor(n_neighbors=10)
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

    def fit(self, model_seq, data_fit_training, data_tag_training):
        # fit
        self.models[model_seq] = self.models[model_seq].fit(X=data_fit_training, y=data_tag_training)

    def plot(self, model_seq, data_fit_training, data_tag_training, x_data_plot_validation, y_data_plot_validation,
             data_fit_validation, data_tag_validation):

        # validation predict
        data_predict_validation = self.models[model_seq].predict(data_fit_validation)
        data_predict_validation = np.reshape(data_predict_validation, (-1, 1))

        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__ + '_validate'
        self.model_mae[model_name] = mean_absolute_error(data_tag_validation, data_predict_validation)

        x_data_plot_validation_size = len(x_data_plot_validation)
        print(model_name)
        for i in range(x_data_plot_validation_size):
            plt.scatter(x_data_plot_validation[i], y_data_plot_validation[i], marker='.',
                        color=self.training_set_plot_color[i])
            plt.scatter(x_data_plot_validation[i], data_predict_validation[i], marker='+',
                        color=self.training_set_plot_color[i])
            # print(str(x_data_plot_validation[i]) + ',' + str(data_predict_validation[i]))

    def validation_plot(self, x_data_base, y_data_base, model_seq, data_file_validation, validation_seq):
        self.base_plot(x_data_base, y_data_base)

        x_data_plot_validation, y_data_plot_validation, y_data_plot_validation_a, data_validation, data_tag_validation = self.load_data(
            data_file=data_file_validation)

        # validation set
        data_validation_predict = self.models[model_seq].predict(data_validation)
        data_validation_predict = np.reshape(data_validation_predict, (-1, 1))
        model_name = str(model_seq) + '_' + self.models[model_seq].__class__.__name__ + '_' + validation_seq
        self.model_mae[model_name] = mean_absolute_error(data_tag_validation, data_validation_predict)
        x_data_plot_validation_size = len(x_data_plot_validation)
        print(model_name)
        for j in range(x_data_plot_validation_size):
            plt.scatter(x_data_plot_validation[j], y_data_plot_validation[j], marker='.',
                        color=self.training_set_plot_color[j])
            plt.scatter(x_data_plot_validation[j], data_validation_predict[j], marker='+',
                        color=self.training_set_plot_color[j])
        output_file_path = self._local_dir + '/output/' + self.models[model_seq].__class__.__name__ + str(
            model_seq) + '_' + validation_seq + '.png'
        plt.savefig(output_file_path)
        plt.show()

    def process(self):
        data_file_training = './input/notch-training-afterlf.csv'
        x_data_plot_training, y_data_plot_training, y_data_plot_training2, data_fit_training, data_tag_training = self.load_data(
            data_file=data_file_training)

        self.model_mae['original'] = mean_absolute_error(data_tag_training, data_fit_training[:, 0])
        x_data_base = x_data_plot_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]
        y_data_base = data_tag_training[
                      self.training_set_base_seq * self.set_size:(self.training_set_base_seq + 1) * self.set_size]

        # pic 1: all temperature and base line
        self.training_plot(x_data_plot_training, y_data_plot_training)
        self.base_plot(x_data_base, y_data_base)
        output_file_path = self._local_dir + '/output/' + 'base.png'
        plt.savefig(output_file_path)
        plt.show()

        self.training_plot(x_data_plot_training, y_data_plot_training2)
        plt.show()

        # pic algorithm:
        for i in range(len(self.models)):
            # print('algorithm:' + str(i))
            self.fit(model_seq=i, data_fit_training=data_fit_training, data_tag_training=data_tag_training)

            data_file_validation1 = '/input/notch-plot-validation1.csv'
            self.validation_plot(x_data_base, y_data_base, i, data_file_validation1, '1')

            data_file_validation2 = '/input/notch-plot-validation2.csv'
            self.validation_plot(x_data_base, y_data_base, i, data_file_validation2, '2')

            data_file_validation3 = '/input/notch-plot-validation3.csv'
            self.validation_plot(x_data_base, y_data_base, i, data_file_validation3, '3')

        # show performance of each algorithm
        print(self.model_mae)


if __name__ == '__main__':
    notch = NotchML()
    notch.process()
