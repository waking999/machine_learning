import os
import pickle

import numpy as np

from AOA.util.Plot import Plot
from AOA.util.Constants import Constants
from AOA.util.DataSet import DataSet
from sklearn.preprocessing import PolynomialFeatures


class PredictNLR:
    def __init__(self, base_file_name, dataset_index, model_name, curve_step, degree):
        self.base_file_name = base_file_name
        self.dataset_index = dataset_index
        self.curve_step = curve_step
        self.model_name = model_name
        self.ploy_reg = PolynomialFeatures(degree=degree)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.dataset_index) + '_' + self.model_name
        model_pickle = open(output_file_path, 'rb')
        self.model = pickle.load(model_pickle)

    def predict(self, xs):
        ys = self.model.predict(xs)
        return ys

    def prepare_valuation(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               dataset_index=self.dataset_index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             dataset_index=self.dataset_index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               dataset_index=self.dataset_index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             dataset_index=self.dataset_index)

        val_x = val_x[:, np.newaxis]
        pred_val_y = self.model.predict(self.ploy_reg.fit_transform(val_x))

        return [train_x, train_y, val_x, val_y, pred_val_y]

    def valuation(self):
        train_x, train_y, val_x, val_y, pred_val_y = self.prepare_valuation()

        train_x = train_x.sort_values()
        train_x = train_x[:, np.newaxis]
        curve_x = self.ploy_reg.fit_transform(train_x)
        curve_y = self.model.predict(curve_x)

        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                  pred_val_y=pred_val_y, curve_x=train_x, curve_y=curve_y, curve_step=self.curve_step,
                  base_file_name=self.base_file_name, image_name=self.model_name + '.png'
                  )


if __name__ == '__main__':
    predict = PredictNLR(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='pf', curve_step=10000,
                         degree=3)
    predict.valuation()
