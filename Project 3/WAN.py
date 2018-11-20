import random

import matplotlib.pyplot as plt
import numpy as np

file = open("train_data_1.txt", "r")
time, energy = ([] for _ in range(2))

for _ in range(16):
    data = file.readline().split(", ")
    time.append(float(data[0]))
    energy.append(float(data[1]))

weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)]

for i in range(16):
    time[i] = (time[i] - np.mean(time)) / np.std(time)

for _ in range(5000):
    for i in range(16):
        net = (time[i] * weights[0]) + weights[1]

        desire = energy[i]
        actual = net
        error = desire - actual
        correction = 0.1 * error

        weights[0] += correction * time[i]
        weights[1] += correction

print(weights)

plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(time, energy)

x = np.array(range(-2, 5))
formula = repr(weights[0]) + '*x+' + repr(weights[1])
y = eval(formula)
plt.plot(x, y, 'k')

plt.show()
