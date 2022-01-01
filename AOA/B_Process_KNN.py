from sklearn.neighbors import KNeighborsRegressor
from DataSet import DataSet
from Util import Util
from Plot import Plot


class Process:
    def __init__(self, base_file_name, index):
        self.base_file_name = base_file_name
        self.index = index

    @staticmethod
    def fit(train_x, train_y):
        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(Util.convert_1d_array_to_2d(train_x), train_y)
        return neigh

    def predict(self, neigh):
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        pred_y = neigh.predict(Util.convert_1d_array_to_2d(val_x))
        return pred_y

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)
        val_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_x',
                                             index=self.index)
        val_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='val_y',
                                             index=self.index)

        neigh = Process.fit(train_x=train_x, train_y=train_y)
        pred_y = self.predict(neigh)

        Plot.plot(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
                  pred_y=pred_y)


if __name__ == '__main__':
    process = Process(base_file_name='E0a.txt_shuffle.csv', index=0)
    process.process()
