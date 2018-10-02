import matplotlib.pyplot as plt
import numpy as np

file = open("data.txt", "r")

height_male, weight_male, height_female, weight_female = ([] for _ in range(4))

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

    plt.show()


graph()

print(height_male)
print(weight_male)
print(height_female)
print(weight_female)
