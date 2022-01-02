import os
import pickle
from DataSet import DataSet
from Constants import Constants
from Util import Util
from Plot import Plot


class Predict:
    def __init__(self, base_file_name, index, model_name):
        self.base_file_name = base_file_name
        self.index = index
        self.model_name = model_name
        self.curve_step = 10000

    def predict(self, xs):
        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.index) + '_' + self.model_name
        knn_pickle = open(output_file_path, 'rb')
        model = pickle.load(knn_pickle)
        ys = model.predict(Util.convert_1d_array_to_2d(xs))
        return ys

    def predict_val(self):
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        pred_val_y = self.predict(val_x)
        return pred_val_y

    def predict_curve(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)

        xs = train_x.append(val_x)
        curve_x = Util.generate_curve_array_2d(xs.sort_values(), self.curve_step)
        curve_y = self.predict(curve_x)
        return [curve_x, curve_y]

    def valuation(self):
        pred_val_y = self.predict_val()
        curve_x, curve_y = self.predict_curve()
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             index=self.index)

        Plot.plot_nlr(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                      pred_val_y=pred_val_y, curve_x=curve_x, curve_y=curve_y, curve_step=self.curve_step,
                      base_file_name=self.base_file_name, image_name=self.model_name + '.png'
                      )
