import os
import numpy as np
import pandas as pd

from DataSet import DataSet
from Util import Util
from Constants import Constants


class PreProces:

    def __init__(self):
        self.max_decimal_place = 10

    def pre_proces(self, input_file_name):
        output_file_name = self.load_shuffle_dataset(input_file_name)
        self.split_train_val_dataset(output_file_name)

    def load_shuffle_dataset(self, input_file_name):
        # load data from input file
        local_dir = os.path.dirname(__file__)
        input_file_path = local_dir + '/' + Constants.DIRECTORY_INPUT + '/' + input_file_name
        df = DataSet.load_dataset(file_path=input_file_path, sep_char='\t', header=None)
        data_array = df.to_numpy()

        # shuffle
        np.random.shuffle(data_array)

        # get decimal number length
        d0f = Util.get_decimal_length(data_array[0][0])

        self.max_decimal_place = d0f

        # save shuffled result to csv formate file
        shuffled_df = pd.DataFrame(data_array)
        output_file_name = input_file_name + '_shuffle.csv'
        output_file_path = local_dir + '/' + Constants.DIRECTORY_WORK + '/' + output_file_name
        DataSet.write_dataset(file_path=output_file_path, df=shuffled_df,
                              sep_char=',', header=None)
        return output_file_name

    def flatten_save(self, data_array, slice_array, base_file_name, array_name):
        array = DataSet.flatten(data_array=data_array, slice_array=slice_array)
        _local_dir = os.path.dirname(__file__)
        output_file_path = _local_dir + '/' + Constants.DIRECTORY_WORK + '/' + base_file_name + '_' + array_name + '.csv'
        np.savetxt(output_file_path, array, delimiter=',', fmt='%.' + str(self.max_decimal_place) + 'f')

    def split_train_val_dataset(self, csv_input_file_name):
        # load from csv format file
        _local_dir = os.path.dirname(__file__)
        input_file_path = _local_dir + '/' + Constants.DIRECTORY_WORK + '/' + csv_input_file_name
        df = DataSet.load_dataset(file_path=input_file_path, sep_char=',', header=None)

        # split data into slices
        x = df[0].values
        y = df[1].values
        num_slice = 5
        x_slices = DataSet.slice_array(x, num_slice=num_slice)
        y_slices = DataSet.slice_array(y, num_slice=num_slice)

        split_ways = [([1, 2, 3, 4], [0])]

        # organise training and valuation data set by different slices combination and save to csv format file
        for index, split_way in enumerate(split_ways):
            self.flatten_save(data_array=x_slices, slice_array=split_way[0],
                              base_file_name=csv_input_file_name, array_name='train_x' + str(index))
            self.flatten_save(data_array=y_slices, slice_array=split_way[0],
                              base_file_name=csv_input_file_name, array_name='train_y' + str(index))
            self.flatten_save(data_array=x_slices, slice_array=split_way[1],
                              base_file_name=csv_input_file_name, array_name='val_x' + str(index))
            self.flatten_save(data_array=y_slices, slice_array=split_way[1],
                              base_file_name=csv_input_file_name, array_name='val_y' + str(index))


if __name__ == '__main__':
    PreProces().pre_proces(input_file_name='E0a.txt')
