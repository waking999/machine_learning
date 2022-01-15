from AOA.util.DataSet import DataSet
from AOA.util.Util import Util


class Predict:
    def __init__(self, base_file_name, dataset_index, curve_step):
        self.base_file_name = base_file_name
        self.dataset_index = dataset_index
        self.curve_step = curve_step

    def predict(self, xs):
        return None

    def predict_val(self, val_x):
        pred_val_y = self.predict(val_x)
        return pred_val_y

    # def predict_curve(self, train_x, val_x):
    #     xs = train_x.append(val_x)
    #     curve_x = Util.generate_curve_array_2d(xs.sort_values(), self.curve_step)
    #     curve_y = self.predict(curve_x)
    #     return [train_x, val_x, curve_x, curve_y]

    def prepare_valuation(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               dataset_index=self.dataset_index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             dataset_index=self.dataset_index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               dataset_index=self.dataset_index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             dataset_index=self.dataset_index)
        # train_x, val_x, curve_x, curve_y = self.predict_curve(train_x=train_x, val_x=val_x)
        pred_val_y = self.predict_val(val_x=val_x)

        return [train_x, train_y, val_x, val_y, pred_val_y]
