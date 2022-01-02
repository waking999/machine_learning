import os
from DataSet import DataSet
from Constants import Constants
from LinearRegression import LinearRegression

from Plot import Plot


class Predict:
    def __init__(self, base_file_name, index):
        self.base_file_name = base_file_name
        self.index = index

        _local_dir = os.path.dirname(__file__)
        input_file_path_wb = _local_dir + '/' + Constants.DIRECTORY_WORK + '/' + base_file_name + '_lr_wb.csv'
        df_wb = DataSet.load_dataset(file_path=input_file_path_wb, sep_char=',', header=None)
        self.w = df_wb[0][0]
        self.b = df_wb[0][1]

    def predict(self, xs):
        ys = LinearRegression.predict(xs, self.w, self.b)
        return ys

    def predict_val(self):
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        pred_val_y = self.predict(val_x)
        return pred_val_y

    def valuation(self):
        pred_val_y = self.predict_val()
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             index=self.index)

        Plot.plot_lr(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                     pred_val_y=pred_val_y, base_file_name=self.base_file_name, image_name='lr.png',
                     w=self.w, b=self.b,
                     line_space_x1=0, line_space_x2=max(train_x), line_space_n=100
                     )


if __name__ == '__main__':
    predict = Predict(base_file_name='E0a.txt_shuffle.csv', index=0)
    predict.valuation()
