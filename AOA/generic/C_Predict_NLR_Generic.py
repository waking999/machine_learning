import os
import pickle
from AOA.util.Constants import Constants
from AOA.util.Util import Util
from AOA.util.Plot import Plot
from AOA.generic.C_Predict_Generic import Predict


class PredictNLR(Predict):
    def __init__(self, base_file_name, dataset_index, model_name, curve_step):
        super().__init__(base_file_name=base_file_name, dataset_index=dataset_index, curve_step=curve_step)
        self.model_name = model_name

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/../' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.dataset_index) + '_' + self.model_name
        model_pickle = open(output_file_path, 'rb')
        self.model = pickle.load(model_pickle)

    def predict(self, xs):
        ys = self.model.predict(Util.convert_1d_array_to_2d(xs))
        return ys

    def valuation(self):
        train_x, train_y, val_x, val_y, pred_val_y, curve_x, curve_y = self.prepare_valuation()

        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                      pred_val_y=pred_val_y, curve_x=curve_x, curve_y=curve_y, curve_step=self.curve_step,
                      base_file_name=self.base_file_name, image_name=self.model_name + '.png'
                      )
