import numpy as np

file = open("data.txt", "w")

height_male = np.random.normal(69, 2.8, 2000)
weight_male = np.random.normal(172, 31.1, 2000)

height_female = np.random.normal(63.6, 2.5, 2000)
weight_female = np.random.normal(143, 32.3, 2000)


def format_data(value):
    return "%.2f" % value


for i in range(2000):
    file.write(format_data(height_male[i]) + ', ' + format_data(weight_male[i]) + ', 0\n')

for i in range(2000):
    file.write(format_data(height_female[i]) + ', ' + format_data(weight_female[i]) + ', 1\n')