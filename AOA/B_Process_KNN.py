from sklearn.neighbors import KNeighborsRegressor
from DataSet import DataSet
from Util import Util
from Constants import Constants
import pickle
import os


class Process:
    def __init__(self, base_file_name, index):
        self.base_file_name = base_file_name
        self.index = index

    @staticmethod
    def fit(train_x, train_y):
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(Util.convert_1d_array_to_2d(train_x), train_y)
        return model

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)

        model = Process.fit(train_x=train_x, train_y=train_y)
        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.index) + '_knn'
        knn_pickle = open(output_file_path, 'wb')
        pickle.dump(model, knn_pickle)


if __name__ == '__main__':
    process = Process(base_file_name='E0a.txt_shuffle.csv', index=0)
    process.process()
