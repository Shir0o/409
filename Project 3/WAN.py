import random
import numpy as np
import matplotlib.pyplot as plt

file = open("train_data_1.txt", "r")
time, energy = ([] for _ in range(2))
weights = []

for _ in range(16):
    data = file.readline().split(", ")
    time.append(float(data[0]))
    energy.append(float(data[1]))

weights = [round(random.uniform(-0.5, 0.5)), round(random.uniform(-0.5, 0.5)),
           round(random.uniform(-0.5, 0.5)), round(random.uniform(-0.5, 0.5))]

for i in range(16):
    net = ((time[i] ** 3) * weights[0]) + ((time[i] ** 2) * weights[1]) + (time[i] * weights[2]) + weights[3]

    desire = energy[i]
    actual = net
    error = desire - actual
    correction = 0.1 * error

    weights[0] += correction * (time[i] ** 3)
    weights[1] += correction * (time[i] ** 2)
    weights[2] += correction * time[i]
    weights[3] += correction

print(weights)

plt.scatter(time, energy)
plt.xlabel('Time')
plt.ylabel('Energy')

x = np.array(range(0, 25))
formula = repr(weights[0]) + '*x**3+' + repr(weights[1]) + '*x**2+' + repr(weights[2]) + '*x+' + repr(weights[3])
print(formula)
y = eval(formula)
plt.plot(x, y)

plt.show()
