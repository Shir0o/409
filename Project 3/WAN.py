import random

import matplotlib.pyplot as plt
import numpy as np


def custom_range(x, y, jump):
    arr = []
    while x < y:
        arr.append(x)
        x += jump

    return arr


def graph():
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.scatter(time, energy)

    x = np.array(custom_range(0, 1.1, 0.01))
    y = weights[0] * (x ** 3) + weights[1] * (x ** 2) + weights[2] * x + weights[3]
    print(x)
    print(y)
    plt.plot(x, y, 'k')

    plt.show()


file = open("train_data_1.txt", "r")
time, energy = ([] for _ in range(2))

for _ in range(16):
    data = file.readline().split(", ")
    time.append(float(data[0]))
    energy.append(float(data[1]))

weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2),
           round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)]
# weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)]

for i in range(16):
    # time[i] = (time[i] - np.mean(time)) / np.std(time)
    # energy[i] = (energy[i] - np.mean(energy)) / np.std(energy)
    time[i] = (time[i] - min(time)) / (max(time) - min(time))
    # energy[i] = (energy[i] - min(energy)) / (max(energy) - min(energy))

print(time)
print(energy)

for _ in range(10000):
    total_error = 0

    for i in range(16):
        net = ((time[i] ** 3) * weights[0]) + ((time[i] ** 2) * weights[1]) + (time[i] * weights[2]) + weights[3]
        # net = (time[i] * weights[0]) + weights[1]

        desire = energy[i]
        actual = net
        error = desire - actual
        total_error += error ** 2
        correction = 0.1 * error

        weights[0] += correction * (time[i] ** 3)
        weights[1] += correction * (time[i] ** 2)
        weights[2] += correction * time[i]
        weights[3] += correction

        # weights[0] += correction * time[i]
        # weights[1] += correction

        # graph()

    print(total_error)

print(weights)

graph()
