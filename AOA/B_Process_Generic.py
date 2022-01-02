# from sklearn.svm import SVR
from DataSet import DataSet
from Util import Util
from Constants import Constants
import pickle
import os


class Process:
    def __init__(self, base_file_name, index, model_name):
        self.base_file_name = base_file_name
        self.index = index
        self.model_name = model_name
        self.model = None

    def fit(self, train_x, train_y):
        self.model.fit(Util.convert_1d_array_to_2d(train_x), train_y)

    def process(self):
        train_x = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_x',
                                               index=self.index)
        train_y = DataSet.load_individual_data(base_file_name=self.base_file_name, array_name='train_y',
                                               index=self.index)

        self.fit(train_x=train_x, train_y=train_y)

        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_MODEL + '/' + self.base_file_name + '_' + str(
            self.index) + '_' + self.model_name
        save_pickle = open(output_file_path, 'wb')
        pickle.dump(self.model, save_pickle)
