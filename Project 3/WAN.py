import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

file = open("train_data_1.txt", "r")
time, energy = ([] for _ in range(2))

for _ in range(16):
    data = file.readline().split(", ")
    time.append(float(data[0]))
    energy.append(float(data[1]))

time = np.array(time)
energy = np.array(energy)

time = time[:, np.newaxis]
energy = energy[:, np.newaxis]

polynomial_features = PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(time)

model = LinearRegression()
model.fit(x_poly, energy)
prediction = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(energy, prediction))
r2 = r2_score(energy, prediction)
print(rmse)
print(r2)

plt.scatter(time, energy, s=10)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(time, prediction), key=sort_axis)
time, prediction = zip(*sorted_zip)
plt.plot(time, prediction, color='m')
plt.show()
