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
import matplotlib

height_male = np.random.normal(70, 4, 2000)
weight_male = np.random.normal(182.9, 40.8, 2000)

height_female = np.random.normal(65, 3.5, 2000)
weight_female = np.random.normal(143, 29, 2000)

true_negative = 0
false_negative = 0
true_positive = 0
false_positive = 0

for i in range(2000):
    calculation = height_male[i] + (0.141 * weight_male[i]) - 92.84
    if calculation > 0:
        true_positive += 1
    elif calculation < 0:
        false_negative += 1

for i in range(2000):
    calculation = height_female[i] + (0.141 * weight_female[i]) - 92.84
    if calculation < 0:
        true_negative += 1
    elif calculation > 0:
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

true_negative = 0
false_negative = 0
true_positive = 0
false_positive = 0

neuron_weights = [0.141, 1, -92.84]

for i in range(1500):
    net = (weight_male[i] * neuron_weights[0]) + (height_male[i] * neuron_weights[1]) + neuron_weights[2]

    desire = 1
    actual = 1 if net > 0 else -1
    correction = 0.3 * (desire - actual)

    neuron_weights[0] += correction * weight_male[i]
    neuron_weights[1] += correction * height_male[i]
    neuron_weights[2] += correction

    net = (weight_female[i] * neuron_weights[0]) + (height_female[i] * neuron_weights[1]) + neuron_weights[2]

    desire = -1
    actual = -1 if net < 0 else 1
    correction = 0.3 * (desire - actual)

    neuron_weights[0] += correction * weight_female[i]
    neuron_weights[1] += correction * height_female[i]
    neuron_weights[2] += correction

for i in range(500):
    net = (weight_male[i] * neuron_weights[0]) + (height_male[i] * neuron_weights[1]) + neuron_weights[2]

    if net > 0:
        true_positive += 1
    elif net < 0:
        false_negative += 1

    net = (weight_female[i] * neuron_weights[0]) + (height_female[i] * neuron_weights[1]) + neuron_weights[2]

    if net < 0:
        true_negative += 1
    elif net > 0:
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

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Height and Weight')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.scatter(weight_female, height_female, s=5, c="r", alpha=0.5, marker=r'o', label="Female")
plt.scatter(weight_male, height_male, s=5, c="b", alpha=0.5, marker=r'o', label="Male")
plt.legend(loc=1)
f, ax = plt.subplots()
ones = []
for x in weight_male:
    ones.append(1)
ax.scatter(height_female, ones, s=10, c="r", alpha=0.1, marker=r'o', label="Female")
ax.scatter(height_male, ones, s=10, c="b", alpha=0.1, marker=r'o', label="Male")
ax.set_title('Height')
plt.xlabel("Height")
plt.ylabel("N/A")

plt.show()
