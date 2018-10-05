import random

import matplotlib.pyplot as plt
import numpy as np

file = open("data.txt", "r")

height_male, weight_male, height_female, weight_female = ([] for _ in range(4))
neuron_weights = []

for _ in range(2000):
    data = file.readline().split(", ")
    height = float(data[0])
    weight = float(data[1])

    # height = (height - 69) / 2.8
    # weight = (weight - 172) / 31.1

    height_male.append(height)
    weight_male.append(weight)

for _ in range(2000):
    data = file.readline().split(", ")
    height = float(data[0])
    weight = float(data[1])

    # height = (height - 63.6) / 2.5
    # weight = (weight - 143) / 32.3

    height_female.append(height)
    weight_female.append(weight)


# height, weight = ([] for _ in range(2))
# for _ in range(2000):
#     data = file.readline().split(", ")
#     height.append(float(data[0]))
#     weight.append(float(data[1]))
#
# height_male = np.array(height)
# weight_male = np.array(weight)
#
# height, weight = ([] for _ in range(2))
# for _ in range(2000):
#     data = file.readline().split(", ")
#     height.append(float(data[0]))
#     weight.append(float(data[1]))
#
# height_female = np.array(height)
# weight_female = np.array(weight)
#
# height_male = (height_male - min(height_male)) / (max(height_male) - min(height_male))
# weight_male = (weight_male - min(weight_male)) / (max(weight_male) - min(weight_male))
#
# height_female = (height_female - min(height_female)) / (max(height_female) - min(height_female))
# weight_female = (weight_female - min(weight_female)) / (max(weight_female) - min(weight_female))


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


def activation(soft, male, net):
    # print(net)
    if soft:
        return np.tanh(0.5 * net)
    else:
        if male:
            return 1 if net >= 0 else -1
        else:
            return -1 if net < 0 else 1


def perceptron(training, iteration, soft):
    global neuron_weights
    neuron_weights = [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2),
                      round(random.uniform(-0.5, 0.5), 2)]
    print(neuron_weights)

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

        # print(total_error)
        # print(count)

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

    print("\n")
    print("True Positive:", true_positive)
    print("False Positive:", false_positive)
    print("True Negative:", true_negative)
    print("False Negative:", false_negative)
    print("Accuracy: ", accuracy)
    print("Error: ", error)
    print("\n")

    graph()


# perceptron(1500, 2000, False)
# perceptron(500, 2000, False)

perceptron(1500, 2000, True)
perceptron(500, 2000, True)
