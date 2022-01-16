import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

x_data_org = np.array([0,
                       1.543209877,
                       3.086419753,
                       4.62962963,
                       6.172839506,
                       7.716049383,
                       9.259259259,
                       10.80246914,
                       12.34567901,
                       13.88888889,
                       15.43209877,
                       16.97530864,
                       18.51851852,
                       20.0617284,
                       21.60493827,
                       22.37654321,
                       23.14814815,
                       23.91975309,
                       24.69135802,
                       25.46296296,
                       26.2345679,
                       27.00617284,
                       27.77777778
                       ])
x_data = x_data_org[..., np.newaxis]
y_data_org = np.array([64.9,
                       64.4,
                       63.3,
                       60,
                       55.9,
                       52.3,
                       48,
                       43.1,
                       38.4,
                       32.8,
                       27.4,
                       21.7,
                       16.5,
                       12.1,
                       8.1,
                       6.3,
                       4.7,
                       3.3,
                       2.5,
                       1.7,
                       1.1,
                       0.9,
                       0.9
                       ])
y_data = y_data_org[..., np.newaxis]
# plt.scatter(x_data, y_data)
# plt.show()

num_x = np.mat(x_data).shape[0]
print(num_x)

X_data = np.concatenate((np.ones((num_x, 1)), x_data), axis=1)


def weights(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix can not do inverse')
        return

    ws = xTx.I * x_mat.T * y_mat
    return ws


ws = weights(X_data, y_data)
print(ws)

x_test = x_data_org
y_test = ws[0, 0] + ws[1, 0] * x_test
plt.plot(x_test, y_test, 'r')
plt.plot(x_data_org, y_data_org, 'b')
plt.show()
