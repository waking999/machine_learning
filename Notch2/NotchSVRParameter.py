import os
import time
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

# search to get the best SVR parameter

class NotchSVRParameter:
    def __init__(self):
        self.min_svr_e = None
        self.num_training_set = 11
        self.set_size = 9
        self.training_set_base_seq = self.num_training_set // 2  # the data set is ordered by temperature ascending, we take the middle one
        self._local_dir = _local_dir = os.path.dirname(__file__)
        self.model_mae = {}

        self.step = 1.5
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

    def load_data(self, data_file):
        df = self.load_dataset(self._local_dir + data_file, sep_char=',', header=None)
        x_data = df.loc[:, [2, 3]].to_numpy()
        y_data = df.loc[:, 4].to_numpy()
        return x_data, y_data

    def mae_svr_parameter_search_on_training_loop(self, x_data_training, y_data_training):

        c_min = self.step ** -10
        c_max = self.step ** 37
        print('c:' + str(c_min) + ' - ' + str(c_max))

        gamma_min = self.step ** -29
        gamma_max = self.step ** 8
        print('gamma:' + str(gamma_min) + ' - ' + str(gamma_max))


        tol_min = self.step ** -42
        tol_max = self.step ** -4
        print('tol:' + str(tol_min) + ' - ' + str(tol_max))

        epsilon_min = self.step ** -17
        epsilon_max = self.step ** 0
        print('epsilon:' + str(epsilon_min) + ' - ' + str(epsilon_max))

        k = 0
        output_file_path = self._local_dir + '/input/' + 'svr_parameter_value_loop.csv'
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
                        model.fit(X=x_data_training, y=y_data_training)

                        # predict
                        y_data_predict = model.predict(x_data_training)
                        y_data_predict = np.reshape(y_data_predict, (-1, 1))

                        mae_name = 'SVR_' + str(C) + '_' + str(gamma) + '_' + str(epsilon) + '_' + str(tol)
                        self.model_mae[mae_name] = mean_absolute_error(y_data_training, y_data_predict)

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
                            if not self.model_svr_mae.empty:
                                self.model_svr_mae.to_csv(output_file_path, index=False, mode='a', header=True)
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

    def mae_svr_parameter_search_on_training_grid(self, x_data_training, y_data_training):
        parameters = [{
            'C': np.logspace(base=self.step, start=-10, stop=37, num=(37 + 10 + 1), endpoint=True),
            'gamma': np.logspace(base=self.step, start=-29, stop=8, num=(8 + 29 + 1), endpoint=True),
            'tol': np.logspace(base=self.step, start=-42, stop=-4, num=(-4 + 42 + 1), endpoint=True),
            'epsilon': np.logspace(base=self.step, start=-17, stop=0, num=(0 + 17 + 1), endpoint=True)
        }]
        #print(parameters)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        clf = GridSearchCV(
            SVR(kernel='rbf', shrinking=True, degree=3, cache_size=200, max_iter=-1),
            param_grid=parameters,
            scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1
        )

        clf.fit(X=x_data_training, y=y_data_training)

        with open('./input/svr_parameter_value_grid.csv', 'w') as f:
            w = csv.DictWriter(f, clf.best_params_.keys())
            w.writeheader()
            w.writerow(clf.best_params_)

        print('clf.best_params_', clf.best_params_)

    def process(self):
        data_file_training = './input/notch-training-afterlf.csv'
        x_data_training, y_data_training = self.load_data(data_file=data_file_training)
        self.model_mae['original'] = mean_absolute_error(y_data_training, x_data_training[:, 0])
        #
        # # search parameter by loop
        # t1 = time.time()
        # self.mae_svr_parameter_search_on_training_loop(x_data_training=x_data_training, y_data_training=y_data_training)
        # t2 = time.time()
        #
        # print('mae parameter search (loop) spends:' + str((t2 - t1)) + 's')


        # search parameter by grid search
        t1 = time.time()
        self.mae_svr_parameter_search_on_training_grid(x_data_training=x_data_training, y_data_training=y_data_training)
        t2 = time.time()

        print('mae parameter search (grid search) spends:' + str((t2 - t1)) + 's')

        print('end')


if __name__ == '__main__':
    notchSVRParameter = NotchSVRParameter()

    notchSVRParameter.process()

    print('end')
