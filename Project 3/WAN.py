"""
Sudhansu Shrestha
Ibrahim Rahman
Yilong Wang

Python version: 3.7
Required library: numpy, matplotlib

Files Included for this project:
1. data.txt : Data sets (including training and testing)
2. WAN.py : Main source code that runs delta learning algorithm on the data

Total error and trained weights are printed in command line.
Graphs are also generated with title and axis labeled.
"""

import random

import matplotlib.pyplot as plt
import numpy as np

patterns = 16
weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2),
           round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)]


def custom_range(x, y, jump):
    arr = []
    while x < y:
        arr.append(x)
        x += jump

    return arr


def graph(polynomial, time, energy, file_name):
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(file_name)
    plt.scatter(time, energy)

    x = np.array(custom_range(0, 1.1, 0.01))
    y = 0
    if polynomial == 3:
        y = weights[0] * (x ** 3) + weights[1] * (x ** 2) + weights[2] * x + weights[3]
    elif polynomial == 2:
        y = weights[0] * (x ** 2) + weights[1] * x + weights[2]
    elif polynomial == 1:
        y = weights[0] * x + weights[1]
    plt.plot(x, y, 'k')

    plt.show()


def reset_weights():
    global weights
    weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2),
               round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)]


def train(file_name, polynomial):
    global weights
    time, energy = ([] for _ in range(2))

    # Read data sets
    file = open(file_name, 'r')
    for _ in range(patterns):
        data = file.readline().split(', ')
        time.append(float(data[0]))
        energy.append(float(data[1]))

    # Normalize
    for i in range(patterns):
        time[i] = (time[i] - min(time)) / (max(time) - min(time))

    total_error = 10
    iterations = 0
    while iterations < 10000 and total_error >= 10:
        total_error = 0
        iterations += 1

        for i in range(patterns):
            # Calculate net
            net = 0
            if polynomial == 3:
                net = ((time[i] ** 3) * weights[0]) + ((time[i] ** 2) * weights[1]) + (time[i] * weights[2]) + weights[
                    3]
            elif polynomial == 2:
                net = ((time[i] ** 2) * weights[0]) + (time[i] * weights[1]) + weights[2]
            elif polynomial == 1:
                net = (time[i] * weights[0]) + weights[1]

            # Calculate error and delta
            desire = energy[i]
            actual = net
            error = desire - actual
            correction = 0.1 * error

            total_error += error ** 2

            # Calculate weight change
            if polynomial == 3:
                weights[0] += correction * (time[i] ** 3)
                weights[1] += correction * (time[i] ** 2)
                weights[2] += correction * time[i]
                weights[3] += correction
            if polynomial == 2:
                weights[0] += correction * (time[i] ** 2)
                weights[1] += correction * time[i]
                weights[2] += correction
            if polynomial == 1:
                weights[0] += correction * time[i]
                weights[1] += correction

    print('Degree ' + str(polynomial) + ' polynomial on ' + file_name)
    print('Total error: :', total_error)
    print('Weights: ', weights)
    print('\n')

    graph(polynomial, time, energy, file_name)


# Training
train('train_data_1.txt', 3)
train('train_data_2.txt', 3)
train('train_data_3.txt', 3)

# Testing
train('test_data_4.txt', 3)

reset_weights()

# Training
train('train_data_1.txt', 2)
train('train_data_2.txt', 2)
train('train_data_3.txt', 2)

# Testing
train('test_data_4.txt', 2)

reset_weights()

# Training
train('train_data_1.txt', 1)
train('train_data_2.txt', 1)
train('train_data_3.txt', 1)

# Testing
train('test_data_4.txt', 1)
