from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from AOA.util.Util import Util
from AOA.util.DataSet import DataSet
from AOA.util.Constants import Constants
import os
import pickle
import numpy as np


class ProcessPF:
    def __init__(self, base_file_name, dataset_index, model_name, degree):
        self.base_file_name = base_file_name
        self.dataset_index = dataset_index
        self.model_name = model_name
        self.model = LinearRegression()
        self.ploy_reg = PolynomialFeatures(degree=degree)

    def fit(self, train_x, train_y):
        self.model.fit(X=train_x, y=train_y)

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               dataset_index=self.dataset_index)
        train_x = train_x[:, np.newaxis]
        train_x_poly = self.ploy_reg.fit_transform(train_x)

        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               dataset_index=self.dataset_index)

        self.fit(train_x=train_x_poly, train_y=train_y)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.dataset_index) + '_' + self.model_name
        save_pickle = open(output_file_path, 'wb')
        pickle.dump(self.model, save_pickle)


if __name__ == '__main__':
    process = ProcessPF(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='pf', degree=3)
    process.process()
