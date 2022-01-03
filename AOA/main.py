import os
import numpy as np
import pandas as pd

from AOA.util.DataSet import DataSet as ds
from LinearRegressionX import LinearRegression


def run():
    # load data from input file
    local_dir = os.path.dirname(__file__)
    file_path = local_dir + '/input/E0a.txt'

    df = ds.load_dataset(file_path=file_path, sep_char='\t', header=None)
    data_array = df.to_numpy()
    # print(data_array)
    np.random.shuffle(data_array)
    # print(data_array)
    shuffled_df = pd.DataFrame(data_array)
    # print(shuffled_df)

    # generate training, valuation
    x = shuffled_df[0]
    y = shuffled_df[1]
    num_slice = 5
    x_slices = ds.slice_array(x, num_slice=num_slice)
    y_slices = ds.slice_array(y, num_slice=num_slice)

    x1_train = ds.flatten(data_array=x_slices, slice_array=[1, 2, 3, 4])
    y1_train = ds.flatten(data_array=y_slices, slice_array=[1, 2, 3, 4])
    x1_val = ds.flatten(data_array=x_slices, slice_array=[0])
    y1_val = ds.flatten(data_array=y_slices, slice_array=[0])

    lr = LinearRegression()
    learning_rate = 0.0003
    w_start = 0  # initial slope guess
    b_start = 0  # initial y-intercept guess
    num_iterations = 300
    lr.run(x_train=x1_train, y_train=y1_train, x_val=x1_val, y_val=y1_val
           , learning_rate=learning_rate, w_start=w_start, b_start=b_start
           , num_iterations=num_iterations)


if __name__ == '__main__':
    run()
