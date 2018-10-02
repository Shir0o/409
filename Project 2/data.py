"""
Sudhansu Shrestha
Ibrahim Rahman
Yilong Wang
9/18/2018

Python version: 3.7
Make sure you have numpy library installed before running it.
This code generates randomized normally distributed data sets and plots it.
"""

import matplotlib.pyplot as plt
import numpy as np

file = open("data.txt", "r")

height_male = np.random.normal(69, 2.8, 20)
weight_male = np.random.normal(172, 31.1, 20)

height_female = np.random.normal(63.6, 2.5, 20)
weight_female = np.random.normal(143, 32.3, 20)

true_negative = 0
false_negative = 0
true_positive = 0
false_positive = 0

# neuron_weights = [0.141, 1, -92.84]
neuron_weights = [1, 1, 1]


def plot_data():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Height and Weight')
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.scatter(weight_female, height_female, s=5, c="r", alpha=0.5, marker=r'o', label="Female")
    plt.scatter(weight_male, height_male, s=5, c="b", alpha=0.5, marker=r'o', label="Male")
    plt.legend(loc=1)

    x = np.array(range(100, 200))
    formula = repr(-(neuron_weights[0] / neuron_weights[1])) + '*x+' + repr(-(neuron_weights[2] / neuron_weights[1]))
    y = eval(formula)
    plt.plot(x, y)

    plt.show()


plot_data()

for i in range(20):
    net = (weight_male[i] * neuron_weights[0]) + (height_male[i] * neuron_weights[1]) + neuron_weights[2]

    desire = 1
    actual = 1 if net >= 0 else -1
    correction = 0.3 * (desire - actual)

    neuron_weights[0] += correction * weight_male[i]
    neuron_weights[1] += correction * height_male[i]
    neuron_weights[2] += correction

    plot_data()

    net = (weight_female[i] * neuron_weights[0]) + (height_female[i] * neuron_weights[1]) + neuron_weights[2]

    desire = -1
    actual = -1 if net < 0 else 1
    correction = 0.3 * (desire - actual)

    neuron_weights[0] += correction * weight_female[i]
    neuron_weights[1] += correction * height_female[i]
    neuron_weights[2] += correction

    plot_data()

print(neuron_weights)

for i in range(5):
    net = (weight_male[i] * neuron_weights[0]) + (height_male[i] * neuron_weights[1]) + neuron_weights[2]

    if net >= 0:
        true_positive += 1
    elif net < 0:
        false_negative += 1

    net = (weight_female[i] * neuron_weights[0]) + (height_female[i] * neuron_weights[1]) + neuron_weights[2]

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

plot_data()
