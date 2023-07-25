import csv
import math
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np

# shuffle the dataset and store them into list of dictionaries.
# Then split dataset into training set and testing set randomly
def randomization():
    examples = []
    examplesTrain = []
    examplesTest = []
    attributes = []
    r = 0
    with open("house_votes_84.csv", 'r') as file:
        allLines = []
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if len(line) == 0:
                continue
            if r == 0:
                for i in line:
                    attributes.append(i)
                r += 1
                continue
            allLines.append(line)
            random.shuffle(allLines)
        file.close()
        for line in allLines:
            dic = {}
            for i in attributes:
                dic[i] = None
            examples.append(dict(zip(dic, line)))
        for j in range(0, 348):
            n = random.choice(examples)
            examplesTrain.append(n)
            examples.remove(n)
        for i in examples:
            examplesTest.append(i)
    return examples, examplesTrain, examplesTest


class TreeNodeInterface():
    def classify(self, example):
        raise NotImplementedError


class DecisionNode(TreeNodeInterface):
    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        self.test_attr_name = test_attr_name
        self.test_attr_threshold = test_attr_threshold
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold)


class LeafNode(TreeNodeInterface):
    def __init__(self, pred_class, pred_class_count, total_count):
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count  # probability of having the class label

    def classify(self, example):
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count,
                                             self.total_count, self.prob)


class DecisionTree:

    def __init__(self, examples, class_name, min_leaf_count=1):
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count
        #self.root = self.learn_tree(examples)  # Information Gain
        self.root = self.learn_tree_Gini(examples)  #Gini

    def learn_tree(self, examples):
        if len(examples) < self.min_leaf_count * 2:
            dic = {}
            for data in range(len(examples)):
                c = examples[data][self.class_name]
                if c not in dic:
                    dic[c] = 1
                else:
                    dic[c] += 1
            dic2 = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
            pred_class = list(dic2.keys())[0]
            predcount = list(dic2.values())[0]
            leaf = LeafNode(pred_class, predcount, len(examples))
            return leaf
        if self.entropy(examples) == 1 or self.entropy(examples) == 0:
            leaf = LeafNode(examples[0][self.class_name], len(examples), len(examples))
            return leaf
        gainmax = -1
        attr = ""
        attr_threshold = -1
        child_lt = []
        child_ge = []
        child_miss = []
        pred_class = ""
        for var in examples[0]:
            newexamples = []
            non = []
            if var == self.class_name:
                break
            for data in range(len(examples)):
                if examples[data][var] is None:
                    non.append(examples[data])
                else:
                    newexamples.append(examples[data])
            if len(newexamples) < self.min_leaf_count * 2:
                continue
            for data in range(len(newexamples)):
                threshold = newexamples[data][var]
                less = []
                greaterorequal = []
                for data1 in range(len(newexamples)):
                    curvalue = newexamples[data1][var]
                    if curvalue < threshold:
                        less.append(newexamples[data1])
                    else:
                        greaterorequal.append(newexamples[data1])
                if len(less) < self.min_leaf_count or len(greaterorequal) < self.min_leaf_count:
                    continue
                lentropy = self.entropy(less)
                gentropy = self.entropy(greaterorequal)
                ent = self.entropy(newexamples)
                gain = ent - ((len(less) / (len(examples))) * lentropy) - (
                            (len(greaterorequal) / (len(examples))) * gentropy)
                if gain > gainmax:
                    gainmax = gain
                    attr = var
                    attr_threshold = threshold
                    child_lt = less
                    child_ge = greaterorequal
                    child_miss = non
        if len(child_lt) < self.min_leaf_count:
            dic = {}
            for data in range(len(examples)):
                c = examples[data][self.class_name]
                if c not in dic:
                    dic[c] = 1
                else:
                    dic[c] += 1
            dic2 = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
            pred_class = list(dic2.keys())[0]
            predcount = list(dic2.values())[0]
            leaf = LeafNode(pred_class, predcount, len(examples))
            return leaf
        lt = self.learn_tree(child_lt)
        ge = self.learn_tree(child_ge)
        if len(child_lt) > len(child_ge):
            node = DecisionNode(attr, attr_threshold, lt, ge, lt)
        else:
            node = DecisionNode(attr, attr_threshold, lt, ge, ge)
        return node

    def learn_tree_Gini(self, examples):
        if len(examples) < self.min_leaf_count * 2:
            dic = {}
            for data in range(len(examples)):
                c = examples[data][self.class_name]
                if c not in dic:
                    dic[c] = 1
                else:
                    dic[c] += 1
            dic2 = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
            pred_class = list(dic2.keys())[0]
            predcount = list(dic2.values())[0]
            leaf = LeafNode(pred_class, predcount, len(examples))
            return leaf
        if self.entropy(examples) == 1 or self.entropy(examples) == 0:
            leaf = LeafNode(examples[0][self.class_name], len(examples), len(examples))
            return leaf
        lowest_Gini = 1
        attr = ""
        attr_threshold = -1
        child_lt = []
        child_ge = []
        child_miss = []
        pred_class = ""
        for var in examples[0]:
            newexamples = []
            non = []
            if var == self.class_name:
                break
            for data in range(len(examples)):
                if examples[data][var] is None:
                    non.append(examples[data])
                else:
                    newexamples.append(examples[data])
            if len(newexamples) < self.min_leaf_count * 2:
                continue
            for data in range(len(newexamples)):
                threshold = newexamples[data][var]
                less = []
                greaterorequal = []
                for data1 in range(len(newexamples)):
                    curvalue = newexamples[data1][var]
                    if curvalue < threshold:
                        less.append(newexamples[data1])
                    else:
                        greaterorequal.append(newexamples[data1])
                if len(less) < self.min_leaf_count or len(greaterorequal) < self.min_leaf_count:
                    continue
                lGini = self.count_Gini(less)
                gGini = self.count_Gini(greaterorequal)
                Gini = ((len(less) / (len(examples))) * lGini) + (
                            (len(greaterorequal) / (len(examples))) * gGini)
                if Gini < lowest_Gini:
                    lowest_Gini = Gini
                    attr = var
                    attr_threshold = threshold
                    child_lt = less
                    child_ge = greaterorequal
                    child_miss = non
        if len(child_lt) < self.min_leaf_count:
            dic = {}
            for data in range(len(examples)):
                c = examples[data][self.class_name]
                if c not in dic:
                    dic[c] = 1
                else:
                    dic[c] += 1
            dic2 = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
            pred_class = list(dic2.keys())[0]
            predcount = list(dic2.values())[0]
            leaf = LeafNode(pred_class, predcount, len(examples))
            return leaf
        lt = self.learn_tree_Gini(child_lt)
        ge = self.learn_tree_Gini(child_ge)
        if len(child_lt) > len(child_ge):
            node = DecisionNode(attr, attr_threshold, lt, ge, lt)
        else:
            node = DecisionNode(attr, attr_threshold, lt, ge, ge)
        return node

    def count_Gini(self, examples):
        Gini = 1
        classes = {}
        probs = {}
        for data in range(len(examples)):
            cur = examples[data][self.class_name]
            if cur not in classes:
                classes[cur] = 1
            else:
                classes[cur] += 1
        for key in classes:
            probs[key] = classes[key] / len(examples)
        for key in probs:
            Gini -= probs[key] * probs[key]
        return Gini


    def entropy(self, examples):
        curentropy = 0
        classes = {}
        probs = {}
        for data in range(len(examples)):
            cur = examples[data][self.class_name]
            if cur not in classes:
                classes[cur] = 1
            else:
                classes[cur] += 1
        for key in classes:
            probs[key] = classes[key] / len(examples)
        for key in probs:
            curentropy -= probs[key] * math.log2(probs[key])
        return curentropy

    def classify(self, example):
        node = self.root
        return node.classify(example)

def test_model(tree, target_examples):
    nume = 0
    denom = 0
    for i in target_examples:
        actual = i["target"]
        pred = tree.classify(i)[0]
        if pred == actual:
            nume += 1
        denom += 1
    return nume/denom


def eachRun():
    r = randomization()
    class_attr_name = 'target'
    min_examples = 10  # minimum number of examples for a leaf node

    examples = r[0]
    train_examples, test_examples = r[1], r[2]

    tree = DecisionTree(train_examples, class_attr_name, min_examples)

    accuracyTrain = test_model(tree, train_examples)
    accuracyTest = test_model(tree, test_examples)
    return accuracyTrain, accuracyTest

accuraciesTrain = []
accuraciesTest = []
for i in range(0, 100):
    result = eachRun()
    accuraciesTrain.append(result[0])
    accuraciesTest.append(result[1])
def graph1():
    fig,ax = plt.subplots(1, 1)
    a = np.array(accuraciesTrain)
    ax.hist(a, bins = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_xticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_title("Mean = " + str(statistics.mean(accuraciesTrain)) + ", std = " + str(statistics.stdev(accuraciesTrain)))
    ax.set_xlabel("(accuracy)")
    ax.set_ylabel("(Accuracy Frequency on Training data)")
    plt.show()

def graph2():
    fig, ax = plt.subplots(1, 1)
    a = np.array(accuraciesTest)
    ax.hist(a, bins=[0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_xticks([0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_title(
        "Mean = " + str(statistics.mean(accuraciesTest)) + ", std = " + str(statistics.stdev(accuraciesTest)))
    ax.set_xlabel("(accuracy)")
    ax.set_ylabel("(Accuracy Frequency on Testing data)")
    plt.show()

graph1()
graph2()

