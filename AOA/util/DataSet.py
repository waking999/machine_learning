import pandas as pd
import numpy as np
import os
from AOA.util.Constants import Constants


class DataSet:

    def __init__(self):
        return

    @staticmethod
    def write_dataset(file_path, df, sep_char, header):
        df.to_csv(file_path, index=False, sep=sep_char, header=header)

    '''
    return pandas.DataFrame
    '''

    @staticmethod
    def load_dataset(file_path, sep_char, header):
        df = pd.read_csv(file_path, sep=sep_char, header=header)
        return df

    '''
    array: [1,2,3,4,5,6,7]
    num_slice: 3
    return: [[1,2],[3,4],[5,6,7]]
    '''

    @staticmethod
    def slice_array(array, num_slice):
        len_array = len(array)
        len_slice = len_array // num_slice
        _rtnArray = []
        for i in range(num_slice):
            start = i * len_slice
            if i < num_slice - 1:
                end = (i + 1) * len_slice
                _rtnArray.append(array[start:end])
            else:
                _rtnArray.append(array[start: len_array])

        return _rtnArray

    '''
    data_array: [[1,2],[3,4],[5,6]]
    slice_array:[0,2]
    return: [1,2,5,6]
    '''

    @staticmethod
    def flatten(data_array, slice_array):
        _rtnArray = []
        for _slice in slice_array:
            _rtnArray.append(data_array[_slice])

        return np.concatenate(_rtnArray)

    @staticmethod
    def load_individual_data(base_file_name, array_name, dataset_index):
        _local_dir = os.path.dirname(__file__)
        input_file_path = _local_dir + '/../' + Constants.DIRECTORY_WORK + '/' + base_file_name + '_' + array_name + str(
            dataset_index) + '.csv'
        df = DataSet.load_dataset(file_path=input_file_path, sep_char=',', header=None)
        return df[0]
