import csv
import math
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np

def get_examples():
    examples = []
    attributes = []
    r = 0
    with open("hw3_cancer.csv", 'r') as file:
        allLines = []
        csv_reader = csv.reader(file)
        for line in csv_reader:
            line = line[0].split('\t') # for wine
            # for house vote
            if len(line) == 0:
                continue
            if r == 0:
                for i in line:
                    attributes.append(i)
                r += 1
            allLines.append(line)
        file.close()
        for line in allLines:
            dic = {}
            for i in attributes:
                dic[i] = None
            examples.append(dict(zip(dic, line)))
    return examples


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

class DecisionNode_cat(TreeNodeInterface):
    def __init__(self, test_attr_name, children_nodes):
        self.test_attr_name = test_attr_name
        self.children_nodes = children_nodes

    def classify(self, example):
        test_val = example[self.test_attr_name]
        return self.children_nodes[test_val].classify(example)

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

    def __init__(self, examples, class_name, min_leaf_count, header):
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count
        self.root = self.learn_tree_num(examples, header)
        #self.root = self.learn_tree_cat(examples, header)

    def learn_tree_num(self, examples, header):
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
        headers = list(header.keys())
        headers.remove(self.class_name)
        for var in findm(headers):
            newexamples = []
            non = []
            if var == self.class_name:
                continue
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
        lt = self.learn_tree_num(child_lt, header)
        ge = self.learn_tree_num(child_ge, header)
        if len(child_lt) > len(child_ge):
            node = DecisionNode(attr, attr_threshold, lt, ge, lt)
        else:
            node = DecisionNode(attr, attr_threshold, lt, ge, ge)
        return node

    def learn_tree_cat(self, examples, header):
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
            leaf = LeafNode(examples[0]["Class"], len(examples), len(examples))
            return leaf
        gainmax = -1
        attr = ""
        final_children = {}
        child_miss = []
        pred_class = ""
        headers = list(header.keys())
        headers.remove(self.class_name)
        for var in findm(headers):
            children = {}
            newexamples = []
            non = []
            if var == self.class_name:
                continue
            for data in range(len(examples)):
                if examples[data][var] is None:
                    non.append(examples[data])
                else:
                    newexamples.append(examples[data])
            if len(newexamples) < self.min_leaf_count * 2:
                continue
            for data in range(len(newexamples)):
                cur = newexamples[data][var]
                if cur not in children:
                    children[cur] = [newexamples[data]]
                else:
                    children[cur].append(newexamples[data])
            entropies = {}
            for key in children:
                entropies[key] = self.entropy(children[key])
            ent = self.entropy(newexamples)
            gain = ent
            for key in children:
                gain -= (len(children[key]) / len(examples)) * entropies[key]
            if gain > gainmax:
                gainmax = gain
                attr = var
                final_children = children
                child_miss = non
        for key in final_children:
            if len(final_children[key]) < self.min_leaf_count:
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
        children_nodes = {}
        for key in final_children:
            children_nodes[key] = self.learn_tree_cat(final_children[key], header)
        node = DecisionNode_cat(attr, children_nodes)
        return node

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

# create bootstraps
def bootstrap(trainingset):
    bs = []
    for i in range(len(trainingset)):
        bs.append(random.choice(trainingset))
    return bs

# find attributes for each node in a random forest
def findm(attributes):
    copyattr = attributes.copy()
    mattributes = []
    for i in range(round(math.sqrt(len(attributes)))):
        cur = random.choice(copyattr)
        mattributes.append(cur)
        copyattr.remove(cur)
    return mattributes

# make the final prediction by selecting the majority choice
def final_predict(predictions):
    prediction = {}
    for i in predictions:
        if i in prediction:
            prediction[i] += 1
        else:
            prediction[i] = 1
    dic2 = dict(sorted(prediction.items(), key=lambda x: x[1], reverse=True))
    return list(dic2.keys())[0]

def cross_validation(ntree, examples, k):
    classes = {}
    copyexamples = examples.copy()
    header = examples[0]
    copyexamples.remove(header)
    for i in copyexamples:
        curclass = i["Class"]
        if curclass not in classes:
            classes[curclass] = [i]
        else:
            classes[curclass].append(i)
    folds = {}
    for n in range(k):
        folds[n] = []
    for key in classes:
        length = round(len(classes[key]) / k)
        for i in range(k - 1):
            for j in range(length):
                cur = random.choice(classes[key])
                folds[i].append(cur)
                classes[key].remove(cur)
        for n in classes[key]:
            folds[k - 1].append(n)
    # finish creating folds
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    for i in folds:
        trainingset = []
        testset = []
        trees = []
        all_predictions = [] # store predictions for all test instances
        for j in folds:
            if i == j:
                testset.extend(folds[i])
            else:
                trainingset.extend(folds[i])
        for tree in range(ntree):
            bstree = bootstrap(trainingset)
            curtree = DecisionTree(bstree, "Class", 77, header)
            trees.append(curtree)
        for j in testset:
            predictions = []
            for curtree in trees:
                predictions.append(curtree.classify(j)[0])
            all_predictions.append(final_predict(predictions))  # take the majority vote
        acc, pre, rec, F1 = cal_performance(all_predictions, testset, classes)
        total_accuracy += acc
        total_precision += pre
        total_recall += rec
        total_F1 += F1
    return total_accuracy/k, total_precision/k, total_recall/k, total_F1/k

def cal_performance(predictions, testset, classes):
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    total_accuracy = 0
    for curclass in classes:
        print(curclass)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(predictions)):
            actual = testset[i]["Class"]
            pred = predictions[i]
            if pred == actual == curclass:
                tp += 1
            elif pred == curclass and actual != curclass:
                fp += 1
            elif pred != curclass and actual == curclass:
                fn += 1
            else:
                tn += 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if tp == 0:
            precision = 0
            recall = 0
            F1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * ((precision * recall) / (precision + recall))
        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision
        total_F1 += F1
        print("tp" + str(tp) + "tn" + str(tn) + "fp" + str(fp) + "fn" + str(fn))
    class_num = len(list(classes.keys()))
    total_accuracy = total_accuracy / class_num
    total_precision = total_precision / class_num
    total_recall = total_recall / class_num
    total_F1 = total_F1 / class_num
    return total_accuracy, total_precision, total_recall, total_F1

examples = get_examples()
result = {}
result[1]= cross_validation(1, examples, 10)
print("--------------")
result[5] = cross_validation(5, examples, 10)
print("--------------")
result[10] = cross_validation(10, examples, 10)
print("--------------")
result[20] = cross_validation(20, examples, 10)
print("--------------")
result[30] = cross_validation(30, examples, 10)
print("--------------")
result[40] = cross_validation(40, examples, 10)
print("--------------")
result[50] = cross_validation(50, examples, 10)
def graph_accuracy():
    y = []
    x = [1, 5, 10, 20, 30, 40, 50]
    for ntree in x:
        y_value = result[ntree][0]
        y.append(y_value)
    plt.plot(x, y)
    plt.title("accuracy as a function of the value of ntree")
    plt.xlabel("value of ntree")
    plt.ylabel("accuracy")
    plt.show()

def graph_precision():
    y = []
    x = [1, 5, 10, 20, 30, 40, 50]
    for ntree in x:
        y_value = result[ntree][1]
        y.append(y_value)
    plt.plot(x, y)
    plt.title("precision as a function of the value of ntree")
    plt.xlabel("value of ntree")
    plt.ylabel("precision")
    plt.show()

def graph_recall():
    y = []
    x = [1, 5, 10, 20, 30, 40, 50]
    for ntree in x:
        y_value = result[ntree][2]
        y.append(y_value)
    plt.plot(x, y)
    plt.title("recall as a function of the value of ntree")
    plt.xlabel("value of ntree")
    plt.ylabel("recall")
    plt.show()

def graph_F1():
    y = []
    x = [1, 5, 10, 20, 30, 40, 50]
    for ntree in x:
        y_value = result[ntree][3]
        y.append(y_value)
    plt.plot(x, y)
    plt.title("F1 as a function of the value of ntree")
    plt.xlabel("value of ntree")
    plt.ylabel("F1")
    plt.show()

graph_accuracy()
graph_precision()
graph_recall()
graph_F1()