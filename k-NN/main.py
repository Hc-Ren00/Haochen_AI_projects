import csv
import math
import random
import matplotlib.pyplot as plt
import statistics

def randomization() :
    allLines = []
    with open('iris.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            if len(line) == 0:
                continue
            allLines.append(line)
        random.shuffle(allLines)  # shuffle the dataset
    file.close()
    train = []
    test = []
    lines = []
    for i in allLines:
        lines.append(i)
    frac = len(train) / len(allLines)
    while frac < 0.8:  # store 80% data in training set randomly
        n = random.choice(lines)
        train.append(n)
        lines.remove(n)
        frac = len(train) / len(allLines)
    for i in lines:  # store the other 20% data in test set
        test.append(i)
    return train, test

def kNN(k, training, testing) :
    A1max = 0
    A1min = 1000
    A2max = 0
    A2min = 1000
    A3max = 0
    A3min = 1000
    A4max = 0
    A4min = 1000
    for i in training:
        if float(i[0]) > float(A1max):
            A1max = i[0]
        if float(i[0]) < float(A1min):
            A1min = i[0]
        if float(i[1]) > float(A2max):
            A2max = i[1]
        if float(i[1]) < float(A2min):
            A2min = i[1]
        if float(i[2]) > float(A3max):
            A3max = i[2]
        if float(i[2]) < float(A3min):
            A3min = i[2]
        if float(i[3]) > float(A4max):
            A4max = i[3]
        if float(i[3]) < float(A4min):
            A4min = i[3]
    for i in training:                          # normalization
        i[0] = str((float(i[0]) - float(A1min)) / float(A1max) - float(A1min))
        i[1] = str((float(i[1]) - float(A2min)) / float(A2max) - float(A2min))
        i[2] = str((float(i[2]) - float(A3min)) / float(A3max) - float(A3min))
        i[3] = str((float(i[3]) - float(A4min)) / float(A4max) - float(A4min))

    predictTrain = []
    predictTest = []
    n1 = 1
    n2 = 1
    for p in training:
        match = []
        dis = []
        for i in training:  # count distance and store nearest k training data
            d = math.sqrt((float(i[0]) - float(p[0])) * (float(i[0]) - float(p[0])) + (float(i[1]) - float(p[1])) * (
                        float(i[1]) - float(p[1])) + (float(i[2]) - float(p[2])) * (float(i[2]) - float(p[2])) + (
                                      float(i[3]) - float(p[3])) * (float(i[3]) - float(p[3])))
            if len(match) < k:
                match.append(i)
                dis.append(d)
            else:
                if d < max(dis):
                    idx = dis.index(max(dis))
                    match.remove(match[idx])
                    dis.remove(dis[idx])
                    match.append(i)
                    dis.append(d)
        vers = 0
        seto = 0
        virg = 0
        for e in match:
            if e[4] == "Iris-setosa":
                seto += 1
            elif e[4] == "Iris-versicolor":
                vers += 1
            elif e[4] == "Iris-virginica":
                virg += 1
        number = max(virg, seto, vers)
        label = None
        if vers == number:  # predict label
            label = "Iris-versicolor"
        elif seto == number:
            label = "Iris-setosa"
        else:
            label = "Iris-virginica"
        predictTrain.append(label)
    for j in testing:
        j[0] = str((float(j[0]) - float(A1min)) / float(A1max) - float(A1min))
        j[1] = str((float(j[1]) - float(A2min)) / float(A2max) - float(A2min))
        j[2] = str((float(j[2]) - float(A3min)) / float(A3max) - float(A3min))
        j[3] = str((float(j[3]) - float(A4min)) / float(A4max) - float(A4min))
        match = []
        dis = []
        for i in training:                       # count distance and store nearest k training data
            d = math.sqrt((float(i[0]) - float(j[0])) * (float(i[0]) - float(j[0])) + (float(i[1]) - float(j[1])) * (float(i[1]) - float(j[1])) + (float(i[2]) - float(j[2])) * (float(i[2]) - float(j[2])) + (float(i[3]) - float(j[3])) * (float(i[3]) - float(j[3])))
            if len(match) < k:
                match.append(i)
                dis.append(d)
            else:
                if d < max(dis):
                    idx = dis.index(max(dis))
                    match.remove(match[idx])
                    dis.remove(dis[idx])
                    match.append(i)
                    dis.append(d)
        vers = 0
        seto = 0
        virg = 0
        for e in match:
            if e[4] == "Iris-setosa":
                seto += 1
            elif e[4] == "Iris-versicolor":
                vers += 1
            elif e[4] == "Iris-virginica":
                virg += 1
        number = max(virg, seto, vers)
        label = None
        if vers == number:                         # predict label
            label = "Iris-versicolor"
        elif seto == number:
            label = "Iris-setosa"
        else:
            label = "Iris-virginica"
        predictTest.append(label)
    return predictTrain, predictTest                              # return the predict labels for all testing sets in a list

def accuracy(prediction , testing):
    nume = 0
    denom = len(testing)
    for i in range(0, len(prediction)):
        if prediction[i] == testing[i][4]:
            nume += 1
    return nume / denom

def eachRun(k):
    accTrain = []
    accTest = []
    sumTrain = 0
    sumTest = 0
    for i in range(0, 20):
        randomData = randomization()
        trainingSet = randomData[0]   # getting training set after shuffle
        testingSet = randomData[1]    # getting testing set after shuffle
        prediction = kNN(k, trainingSet, testingSet)
        accuracyTrain = accuracy(prediction[0], trainingSet)
        accuracyTest = accuracy(prediction[1], testingSet)
        accTrain.append(accuracyTrain)
        accTest.append(accuracyTest)
    stdTrain = statistics.stdev(accTrain)
    stdTest = statistics.stdev(accTest)
    for i in accTrain:
        sumTrain += i
    for i in accTest:
        sumTest += i
    return sumTrain/20, sumTest/20, stdTrain, stdTest
accuraciesTrain = []
accuraciesTest = []
k = []
stdTrain = []
stdTest = []
for i in range(0, 26):
    k.append(i*2 + 1)
for e in k:
    print("yes")
    result = eachRun(e)
    accuraciesTrain.append(result[0])
    accuraciesTest.append(result[1])
    stdTrain.append(result[2])
    stdTest.append(result[3])

def graph1():
    x1 = k
    y1 = accuraciesTrain
    plt.plot(x1, y1, label = "")
    plt.xlabel("(value of k)")
    plt.ylabel("(Accuracy over training data")
    plt.errorbar(x1, y1, yerr=stdTrain, fmt='-o')
    plt.show()

def graph2():
    x1 = k
    y1 = accuraciesTest
    plt.plot(x1, y1, label="")
    plt.xlabel("(value of k)")
    plt.ylabel("(Accuracy over testing data")
    plt.errorbar(x1, y1, yerr=stdTest, fmt='-o')
    plt.show()
graph2()