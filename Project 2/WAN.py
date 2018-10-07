"""
Sudhansu Shrestha
Ibrahim Rahman
Yilong Wang
10/07/2018

Python version: 3.7
Make sure you have numpy library installed before running it.

Files Included for this project:
1. data.txt : This is the dataset generated from Project 1
2. WAN.py : This is the main source code, that trains 75 and 25
            percent of data for hard and soft activation functions
            accordingly.

NOTE: The graphs are generated simultaneously, so close one graph
in order to get the next graph.

Accuracy and error rates are printed in command line.
"""
import random

import matplotlib.pyplot as plt
import numpy as np

"File open data.txt"
file = open("data.txt", "r")

"Initialize arrays and neuron weight"
height_male, weight_male, height_female, weight_female = ([] for _ in range(4))
neuron_weights = []

"Read from data.txt"
for _ in range(2000):
    data = file.readline().split(", ")
    height = float(data[0])
    weight = float(data[1])

    height_male.append(height)
    weight_male.append(weight)

for _ in range(2000):
    data = file.readline().split(", ")
    height = float(data[0])
    weight = float(data[1])

    height_female.append(height)
    weight_female.append(weight)


"This struct graphs the data set and each linear seperator"
def graph():
    plt.scatter(height_male, weight_male, s=5, c='b', alpha=0.5, marker=r'o', label='Male')
    plt.scatter(height_female, weight_female, s=5, c='r', alpha=0.5, marker=r'o', label='Female')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend()

    x = np.array(range(50, 80))
    formula = repr(-(neuron_weights[0] / neuron_weights[1])) + '*x+' + repr(-(neuron_weights[2] / neuron_weights[1]))
    y = eval(formula)
    plt.plot(x, y, 'k', alpha=0.3)

    plt.show()


"Controller strct for soft or hard activation function"
def activation(soft, male, net):
    if soft:
        return np.tanh(0.5 * net)
    else:
        if male:
            return 1 if net >= 0 else -1
        else:
            return -1 if net < 0 else 1


"Neuron training"
def perceptron(training, iteration, soft):
    global neuron_weights
    neuron_weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2),
                      round(random.uniform(-0.5, 0.5), 2)]

    true_negative, false_negative, true_positive, false_positive = (0 for _ in range(4))
    total_error = 1
    count = 0

    print("Training for " + str(training) + " training data...")

    while total_error > 10 ** -5 and count < iteration:
        total_error = 0
        count += 1

        for i in range(training):
            net = (height_male[i] * neuron_weights[0]) + (weight_male[i] * neuron_weights[1]) + neuron_weights[2]

            desire = 1
            actual = activation(soft, True, net)
            error = desire - actual
            correction = 0.1 * error

            neuron_weights[0] += correction * height_male[i]
            neuron_weights[1] += correction * weight_male[i]
            neuron_weights[2] += correction

            total_error += 1 / 4000 if net < 0 else 0

            net = (height_female[i] * neuron_weights[0]) + (weight_female[i] * neuron_weights[1]) + neuron_weights[2]

            desire = -1
            actual = activation(soft, False, net)
            error = desire - actual
            correction = 0.1 * error

            neuron_weights[0] += correction * height_female[i]
            neuron_weights[1] += correction * weight_female[i]
            neuron_weights[2] += correction

            total_error += 1 / 4000 if net >= 0 else 0

    for i in range(training + 1, 2000):
        net = (height_male[i] * neuron_weights[0]) + (weight_male[i] * neuron_weights[1]) + neuron_weights[2]

        if net >= 0:
            true_positive += 1
        elif net < 0:
            false_negative += 1

        net = (height_female[i] * neuron_weights[0]) + (weight_female[i] * neuron_weights[1]) + neuron_weights[2]

        if net < 0:
            true_negative += 1
        elif net >= 0:
            false_positive += 1

    accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
    error = 1 - accuracy

    print("")
    print("True Positive:", true_positive)
    print("False Positive:", false_positive)
    print("True Negative:", true_negative)
    print("False Negative:", false_negative)
    print("Accuracy: ", accuracy)
    print("Error: ", error)
    print("")

    graph()



print("Hard activation:")
perceptron(1500, 2500, False)
perceptron(500, 2500, False)

print("Soft activation:")
perceptron(1500, 2500, True)
perceptron(500, 2500, True)
