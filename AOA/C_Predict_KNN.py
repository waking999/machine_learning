import os
import pickle
from DataSet import DataSet
from Constants import Constants
from Util import Util

from Plot import Plot


class Predict:
    def __init__(self, base_file_name, index):
        self.base_file_name = base_file_name
        self.index = index

    def predict(self, xs):
        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.index) + '_knn'
        knn_pickle = open(output_file_path, 'rb')
        neigh = pickle.load(knn_pickle)
        ys = neigh.predict(Util.convert_1d_array_to_2d(xs))
        return ys

    def predict_val(self):
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        pred_val_y = self.predict(val_x)
        return pred_val_y

    def predict_train(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)

        xs = train_x.append(val_x)
        train_x_sort = xs.sort_values()
        pred_train_y = self.predict(train_x_sort)
        return [train_x_sort, pred_train_y]

    def valuation(self):
        pred_val_y = self.predict_val()
        train_x_sort, pred_train_y = self.predict_train()
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             index=self.index)

        Plot.plot_knn(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                      pred_val_y=pred_val_y, train_x_sort=train_x_sort, pred_train_y=pred_train_y,
                      base_file_name=self.base_file_name, image_name='knn.png'
                      )


if __name__ == '__main__':
    predict = Predict(base_file_name='E0a.txt_shuffle.csv', index=0)
    predict.valuation()
