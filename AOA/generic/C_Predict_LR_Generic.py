import os
from AOA.util.DataSet import DataSet
from AOA.util.Constants import Constants
from AOA.util.Plot import Plot
from AOA.generic.C_Predict_Generic import Predict


class PredictLR(Predict):
    def __init__(self, base_file_name, dataset_index, wb_file_suffix, linear_regression_instance, curve_step):
        super().__init__(base_file_name=base_file_name, dataset_index=dataset_index, curve_step=curve_step)

        self.wb_file_suffix = wb_file_suffix
        self.linear_regression_instance = linear_regression_instance

        _local_dir = os.path.dirname(__file__)
        input_file_path_wb = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + base_file_name + '_' + self.wb_file_suffix
        df_wb = DataSet.load_dataset(file_path=input_file_path_wb, sep_char=',', header=None)
        self.w = df_wb[0][0]
        self.b = df_wb[0][1]

    def predict(self, xs):
        ys = self.linear_regression_instance.predict(xs, self.w, self.b)
        return ys

    def valuation(self):
        train_x, train_y, val_x, val_y, pred_val_y, curve_x, curve_y = self.prepare_valuation()

        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                  curve_x=curve_x, curve_y=curve_y, curve_step=self.curve_step,
                  pred_val_y=pred_val_y, base_file_name=self.base_file_name, image_name=self.wb_file_suffix + '.png'
                  )
