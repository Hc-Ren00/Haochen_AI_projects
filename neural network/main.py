import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt
import neural_network as nn
import random_forest as rf
import decision_tree as dt

def read_in_file(filename):
    examples = []
    class_index = -1
    with open(filename, mode = 'r') as file:
        csvfile = csv.reader(file)
        for line in csvfile:
            if filename == 'parkinsons.csv':
                examples.append(line)
                class_index = 22
            elif filename == 'titanic.csv':
                examples.append(line)
                class_index = 0
            elif filename == 'hw3_cancer.csv':
                newline = line[0].split('\t')
                examples.append(newline)
                class_index = 9
            else:
                examples.append(line)
                calss_index = 16
        header = examples.pop(0)
    file.close()
    return examples, class_index, header

def cat_or_num(var):
    if var == 'Pclass':
        return 'cat'
    if var == 'Sex':
        return 'cat'
    if var == 'Age':
        return 'num'
    if var == 'Siblings/Spouses Aboard':
        return 'num'
    if var == 'Parents/Children Aboar':
        return 'num'
    if var == 'Fare':
        return 'num'
    else:
        return False

def normalization(examples, filename, header):
    if filename == 'titanic.csv':
        header.pop(0)
        remove_index = header.index('Name')
        change_index = header.index('Sex')
        for i in examples:
            change = i[change_index]
            if change == 'female':
                i[change_index] = 1
            else:
                i[change_index] = 0
            i.pop(remove_index)
    minmax_values = {}
    for i in range(len(examples[0])):
        minmax_values[i] = [10000000, -10000000]
    for instance in examples:
        for j in range(len(instance)):
            instance[j] = float(instance[j])
            if instance[j] < minmax_values[j][0]:
                minmax_values[j][0] = instance[j]
            if instance[j] > minmax_values[j][1]:
                minmax_values[j][1] = instance[j]
    for instance in examples:
        for j in range(len(instance)):
            instance[j] = (float(instance[j]) - float(minmax_values[j][0])) / (float(minmax_values[j][1]) - float(minmax_values[j][0]))
    return examples

def cross_validation(examples, k, class_index, structure, filename, header):
    classes = {}
    instances = []
    copyexamples = []
    for i in examples:
        copyexamples.append(i.copy())
    for i in range(len(copyexamples)):
        classname = copyexamples[i].pop(class_index)
    if structure != 'random_forest' and structure != 'decision_tree':
        copyexamples = normalization(copyexamples, filename, header)
    for i in range(len(examples)):
        curclass = examples[i][class_index]
        if curclass not in classes:
            classes[curclass] = [(copyexamples[i], curclass)]
        else:
            classes[curclass].append((copyexamples[i], curclass))
    folds = {}
    for n in range(k):
        folds[n] = []
    for key in classes:
        length = math.floor(len(classes[key]) / k)
        for i in range(k - 1):
            for j in range(length):
                cur = random.choice(classes[key])
                folds[i].append(cur)
                classes[key].remove(cur)
        for n in classes[key]:
            folds[k - 1].append(n)
    # finish creating folds
    return folds, classes

examples, class_index, header = read_in_file("parkinsons.csv")
folds, classes = cross_validation(examples, 10, class_index, 'neural_network', 'parkinsons.csv', header)
#print("random_forest with ntree value = 10:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 10, classes, 10))
#print("random_forest with ntree value = 15:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 15, classes, 10))
#print("random_forest with ntree value = 20:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 20, classes, 10))
print("leaf node should have at least 15 instances:")
print(dt.run_dt(dt.rebuild_examples(folds, header), header, class_index, classes, 10, 15))
print("leaf node should have at least 5 instances:")
print(dt.run_dt(dt.rebuild_examples(folds, header), header, class_index, classes, 10, 5))
print("leaf node should have at least 25 instances:")
print(dt.run_dt(dt.rebuild_examples(folds, header), header, class_index, classes, 10, 25))
#print("neural network: 2 hidden-layers, first layer has 8 neurons and second layer has 6 neurons, with lambda = 0.25")
#print(nn.run_nn(folds, 10, class_index,2, [8, 6], 0.25, None, classes))
#print("neural network: 3 hidden-layers, first layer has 5 neurons, second layer has 3 neurons, and third layer has 4 neurons, with lambda = 0.25")
#print(nn.run_nn(folds, 10, class_index,3, [5, 3, 4], 0.25, None, classes))
#print("neural network: 2 hidden-layers, first layer has 8 neurons and second layer has 6 neurons, with lambda = 0.6")
#print(nn.run_nn(folds, 10, class_index,2, [8, 6], 0.6, None, classes))
#rf.graph_accuracy(rf.rebuild_examples(folds, header), header, class_index, classes, 10)
#dt.graph1(dt.rebuild_examples(folds, header), header, class_index, classes, 10, 5)
#nn.learning_curve('parkinsons.csv', examples, class_index, header)
#print("random_forest with ntree value = 10:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 10, classes, 10))
#print("random_forest with ntree value = 15:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 15, classes, 10))
#print("random_forest with ntree value = 20:")
#print(rf.run_rf(rf.rebuild_examples(folds, header), header, class_index, 20, classes, 10))