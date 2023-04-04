import numpy as np
import csv


# function for loading data into a numpy array and a string
def dataloader(filename):
    alldata = []
    with open(filename) as f:
        reader = csv.reader(f)
        while True:
            try:
                line = next(reader)
                alldata.append(list(line))
            except StopIteration:
                print('Finished loading data')
                break

    dataforperceptron = []

    for singleData in alldata:
        dataforperceptron.append([np.array(singleData[:-1]).astype(np.float64), singleData[-1]])

    return dataforperceptron


data = dataloader('perceptron.data')
test = dataloader('perceptron.test.data')

weights = np.random.rand(len(data[0][0]))

theta, alfa, acceptable_accuracy, accuracy, epoch = 0, 0.002, 0.97, 0, 0

# learning
while accuracy < acceptable_accuracy:
    correct = 0
    for i in data:
        print(np.dot(weights, i[0]) - theta)
        if np.dot(weights, i[0]) - theta >= 0:
            print(f'{i[1]} as 1 (Iris-versicolor)')
            if i[1] == 'Iris-versicolor':
                correct += 1
            else:
                weights -= alfa*i[0]
                theta += alfa
        else:
            print(f'{i[1]} as 0 (not Iris-veriscolor)')
            if i[1] != 'Iris-versicolor':
                correct += 1
            else:
                weights += alfa*i[0]
                theta -= alfa
        epoch += 1
    accuracy = correct / len(data)

print(f'accuracy was {accuracy}')
print(f'perceptron took {epoch} epochs')

wait = input('\npress enter to continue\n')

# testing
accuracy = 0
for i in test:
    if np.dot(weights, i[0]) - theta >= 0:
        if i[1] == 'Iris-versicolor':
            print(f'\033[32m{i[1]} classified as Iris-versicolor\033[0m')
            accuracy += 1
        else:
            print(f'\033[31m{i[1]} classified as Iris-versicolor\033[0m')
    else:
        if i[1] != 'Iris-versicolor':
            print(f'\033[32m{i[1]} not classified as Iris-versicolor\033[0m')
            accuracy += 1
        else:
            print(f'\033[31m{i[1]} not classified as Iris-versicolor\033[0m')

accuracy = accuracy / len(test)
print(f'accuracy was {accuracy}')

testing = input('do you want to test a vector? [y/n]\n')
while testing == 'y':
    vector = []
    for i in range(0, len(test[0][0])):
        vector.append(float(input('insert a double and press enter\n')))
    if np.dot(vector, weights) - theta >= 0:
        print('inserted values belong to an Iris-versicolor')
    else:
        print('inserted values do not belong to an Iris-versicolor')
    testing = input('do you want to test a vector? [y/n]\n')
