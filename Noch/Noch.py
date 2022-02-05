import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class Noch:
    def __init__(self):
        self._local_dir = _local_dir = os.path.dirname(__file__)
        self.group_size = None
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        # https://matplotlib.org/stable/api/markers_api.html
        self.decision_tree = {"model": DecisionTreeRegressor(max_depth=10), "marker": '+', "color": 'lightpink'}
        self.knn = {"model": KNeighborsRegressor(n_neighbors=2), "marker": 'x', "color": 'crimson'}
        # self.mlp = {"model": MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto',
        #                                   learning_rate='constant',
        #                                   learning_rate_init=0.0001, power_t=0.5,
        #                                   max_iter=200000, momentum=0.9, shuffle=True, random_state=None,
        #                                   nesterovs_momentum=True, solver='adam', tol=0.0001,
        #                                   n_iter_no_change=10, validation_fraction=0.1, verbose=False,
        #                                   warm_start=True, hidden_layer_sizes=(512, 256, 128, 64, 32)
        #                                   ), "marker": '1', "color": 'gold'}
        # self.svr = {"model": SVR(C=50.0, cache_size=200, degree=100, epsilon=0.001,
        #                  gamma=0.1, kernel='rbf',
        #                  max_iter=-1, shrinking=True, tol=0.001, verbose=False), "marker": '2', "color": 'orange'}

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

    @staticmethod
    def weights(x_arr, y_arr):
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr)
        xTx = x_mat.T * x_mat
        if np.linalg.det(xTx) == 0.0:
            print('This matrix can not do inverse')
            return

        _ws = xTx.I * x_mat.T * y_mat.T
        return _ws

    @staticmethod
    def onem_2_mone(arr):
        return arr.reshape(-1, 1)

    def linear_fitting(self, base_data_file):
        df = self.load_dataset(self._local_dir + base_data_file, sep_char=',', header=None)

        x_data_base = df.loc[:, 1].to_numpy()
        y_data_base = df.loc[:, 2].to_numpy()

        self.group_size = np.mat(x_data_base).shape[1]
        num_ones = np.ones((self.group_size, 1))
        X_data = np.concatenate((num_ones, self.onem_2_mone(x_data_base)), axis=1)
        ws = self.weights(X_data, y_data_base)
        y_data_linearfitting = ws.T * X_data.T

        # linear fitting
        # plt.scatter(x_data_base, x_data_base, marker='.', c='lightgreen')
        plt.plot(x_data_base, y_data_linearfitting.T, c='lightgreen')

    def model(self, x_data_training, y_data_training, fix_temperature, model_dict):
        x_data_training_notemp = x_data_training.copy()
        x_data_training_notemp[:, 0] = fix_temperature
        model_dict["model"].fit(X=x_data_training, y=y_data_training)
        y_data_model1 = model_dict["model"].predict(x_data_training_notemp[:self.group_size])
        plt.scatter(x_data_training_notemp[:self.group_size, 1], y_data_model1, marker=model_dict["marker"],
                    c=model_dict["color"])

    def proces(self):
        self.linear_fitting(base_data_file='/input/dataset1.csv')

        # load training
        df = self.load_dataset(self._local_dir + '/input/training.csv', sep_char=',', header=None)
        x_data_training = df.loc[:, 0:1].to_numpy()
        y_data_training = df.loc[:, 2].to_numpy()

        x_data_training_1 = x_data_training[:self.group_size, 1]
        y_data_training_1 = y_data_training[:self.group_size]
        plt.scatter(x_data_training_1, y_data_training_1, marker='.', c='lightgreen')

        x_data_training_2 = x_data_training[self.group_size:2 * self.group_size, 1]
        y_data_training_2 = y_data_training[self.group_size:2 * self.group_size]
        plt.scatter(x_data_training_2, y_data_training_2, marker='.', c='lightblue')

        # decision_tree
        self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
                   model_dict=self.decision_tree)

        # knn
        self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
                   model_dict=self.knn)

        # # mlp
        # self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
        #            model_dict=self.mlp)

        # svr
        # self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
        #            model_dict=self.svr)

        # xgb
        self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
                   model_dict=self.xgb)

        # lr
        self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
                   model_dict=self.lr)

        # rf
        self.model(x_data_training=x_data_training, y_data_training=y_data_training, fix_temperature=25.3,
                   model_dict=self.rf)

        plt.show()


if __name__ == '__main__':
    noch = Noch()
    noch.proces()
