import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x_data = np.array([0,
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
                   27.77777778])
y_data = np.array([64.9,
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
                   0.9])

# plt.scatter(x_data, y_data)
# plt.show()

x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]

model = LinearRegression()
model.fit(x_data, y_data)

# plt.plot(x_data, y_data, 'b.')
# plt.plot(x_data, model.predict(x_data), 'r')
# plt.show()

ploy_reg = PolynomialFeatures(degree=10)
x_poly = ploy_reg.fit_transform(x_data)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(x_poly), c='r')
plt.show()
