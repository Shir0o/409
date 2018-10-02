import matplotlib.pyplot as plt
import numpy as np

file = open("data.txt", "r")

height_male, weight_male, height_female, weight_female = ([] for _ in range(4))
true_negative, false_negative, true_positive, false_positive = (0 for _ in range(4))
total_error = 1

# neuron_weights = [0.141, 1, -92.84]
neuron_weights = [1, 1, 1]

for _ in range(2000):
    data = file.readline().split(", ")
    height_male.append(float(data[0]))
    weight_male.append(float(data[1]))

for _ in range(2000):
    data = file.readline().split(", ")
    height_female.append(float(data[0]))
    weight_female.append(float(data[1]))


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


while total_error > 0.2:
    total_error = 0

    for i in range(1500):
        net = (height_male[i] * neuron_weights[0]) + (weight_male[i] * neuron_weights[1]) + neuron_weights[2]

        desire = 1
        actual = 1 if net >= 0 else -1
        error = desire - actual
        correction = 0.5 * error

        neuron_weights[0] += correction * height_male[i]
        neuron_weights[1] += correction * weight_male[i]
        neuron_weights[2] += correction

        total_error += 1 / 4000 if net < 0 else 0
        # graph()

        net = (height_female[i] * neuron_weights[0]) + (weight_female[i] * neuron_weights[1]) + neuron_weights[2]

        desire = -1
        actual = -1 if net < 0 else 1
        error = desire - actual
        correction = 0.5 * error

        neuron_weights[0] += correction * height_female[i]
        neuron_weights[1] += correction * weight_female[i]
        neuron_weights[2] += correction

        total_error += 1 / 4000 if net >= 0 else 0
        # graph()

    print(total_error)

print(neuron_weights)

for i in range(500):
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
