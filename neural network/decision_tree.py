import random
import math
import matplotlib.pyplot as plt
import statistics
import numpy as np

def rebuild_examples(folds, header):
    #name_index = header.index('Name')
    new_folds = {}
    for n in folds:
        new_folds[n] = []
        for e in folds[n]:
            instance = {}
            lst = []
            lst = e[0].copy()
            lst.insert(0, e[1])
            #lst.pop(name_index) # remove Name from the instance since it is irrelevant
            N = 0
            for h in header:
                if h == 'Name':
                    continue
                else:
                    instance[h] = lst[N]
                    N += 1
            new_folds[n].append(instance)
    return new_folds

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
        return 'num'

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
        elif float(test_val) < self.test_attr_threshold:
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

    def __init__(self, examples, class_name, min_leaf_count, header, class_index):
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count
        self.root = self.learn_tree(examples, header, class_index)

    def learn_tree(self, examples, header, class_index):
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
            leaf = LeafNode(examples[0][header[class_index]], len(examples), len(examples))
            return leaf
        attr_threshold = -1
        child_lt = []
        child_ge = []
        gainmax = -1
        attr = ""
        final_children = {}
        child_miss = []
        pred_class = ""
        headers = list(header)
        headers.remove(self.class_name)
        cat = False
        for var in headers:
            if var == 'Name':
                continue
            if cat_or_num(var) == 'cat':
                children = {}
                newexamples = []
                non = []
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
                    cat = True
            else:
                newexamples = []
                non = []
                for data in range(len(examples)):
                    if examples[data][var] is None:
                        non.append(examples[data])
                    else:
                        newexamples.append(examples[data])
                if len(newexamples) < self.min_leaf_count * 2:
                    continue
                for data in range(len(newexamples)):
                    threshold = float(newexamples[data][var])
                    less = []
                    greaterorequal = []
                    for data1 in range(len(newexamples)):
                        curvalue = float(newexamples[data1][var])
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
                        cat = False
        if cat is True:
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
                children_nodes[key] = self.learn_tree(final_children[key], header, class_index)
            node = DecisionNode_cat(attr, children_nodes)
            return node
        else:
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
            lt = self.learn_tree(child_lt, header, class_index)
            ge = self.learn_tree(child_ge, header, class_index)
            if len(child_lt) > len(child_ge):
                node = DecisionNode(attr, attr_threshold, lt, ge, lt)
            else:
                node = DecisionNode(attr, attr_threshold, lt, ge, ge)
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

def cal_performance(predictions, testset, classes, header, class_index):
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    total_accuracy = 0
    for curclass in classes:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(predictions)):
            actual = testset[i][header[class_index]]
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
        if (fp + fn) == 0:
            precision = 1
            recall = 1
            F1 = 1
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * ((precision * recall) / (precision + recall))
        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision
        total_F1 += F1
    class_num = len(list(classes.keys()))
    total_accuracy = total_accuracy / class_num
    total_precision = total_precision / class_num
    total_recall = total_recall / class_num
    total_F1 = total_F1 / class_num
    return total_accuracy, total_precision, total_recall, total_F1

def run_dt(folds, header, class_index, classes, k, leaf_num):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    for i in folds:
        trainingset = []
        testset = []
        trees = []
        all_predictions = []  # store predictions for all test instances
        for j in folds:
            if i == j:
                testset.extend(folds[i])
            else:
                trainingset.extend(folds[i])
        curtree = DecisionTree(trainingset, header[class_index], leaf_num, header, class_index)
        for j in testset:
            all_predictions.append(curtree.classify(j)[0])
        acc, pre, rec, F1 = cal_performance(all_predictions, testset, classes, header, class_index)
        total_accuracy += acc
        total_precision += pre
        total_recall += rec
        total_F1 += F1
    return total_accuracy / k, total_precision / k, total_recall / k, total_F1 / k

def graph1(folds, header, class_index, classes, k, leaf_num):
    accuraciesTrain = []
    for i in range(0, 10):
        result = run_dt(folds, header, class_index, classes, k, leaf_num)
        print(result)
        accuraciesTrain.append(result[0])
    fig,ax = plt.subplots(1, 1)
    a = np.array(accuraciesTrain)
    ax.hist(a, bins = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_xticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_title("Mean = " + str(statistics.mean(accuraciesTrain)) + ", std = " + str(statistics.stdev(accuraciesTrain)))
    ax.set_xlabel("(accuracy)")
    ax.set_ylabel("(Accuracy Frequency on Training data)")
    plt.show()



