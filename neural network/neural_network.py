import random
import numpy as np
import math
import matplotlib.pyplot as plt

def normalization(examples, header, class_index):
    header.pop(class_index)
    #remove_index = header.index('Name')
    #change_index = header.index('Sex')
    #for i in examples:
        #if i[change_index] == 'male':
            #i[change_index] = 0
        #else:
            #i[change_index] = 1
        #i.pop(remove_index)
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

def forward_propagation(layer_num, neuron_list, weights_list, instance, classes_num):
    pred = []
    a_list = []
    if weights_list is None:
        weights_list = []
        for i in range(layer_num + 1):
            curweight = []
            if i == layer_num:
                for m in range(classes_num):
                    weights = []
                    pre_layer_neuron_num = neuron_list[i - 1] + 1
                    for n in range(pre_layer_neuron_num):
                        weights.append(random.uniform(-1.0, 1.0))
                    curweight.append(weights)
            else:
                for m in range(neuron_list[i]):
                    weights = []
                    pre_layer_neuron_num = 0
                    if i == 0:
                        pre_layer_neuron_num = len(instance[0]) + 1
                    else:
                        pre_layer_neuron_num = neuron_list[i-1] + 1
                    for n in range(pre_layer_neuron_num):
                        weights.append(random.uniform(-1.0, 1.0))
                    curweight.append(weights)
            weights_list.append(curweight)
    #print(weights_list)
    a1 = []
    a1.append([1])
    for i in instance[0]:
        a1.append([i])
    a = a1.copy()
    n = 1
    #print("a" + str(n) + ": " + str(a1))
    a_list.append(a)
    for layer in range(layer_num):
        n += 1
        z = np.dot(np.array(weights_list[layer]), np.array(a))
        #print("z" + str(n) + ": " + str(z))
        a = []
        a.append([1])
        for i in z:
            a.append([1 / (1 + math.exp(-i))])
        a_list.append(a)
        #print("a" + str(n) + ": " + str(a))
    n += 1
    z = np.dot(weights_list[-1], a)
    #print("z" + str(n) + ": " + str(z))
    a = []
    for i in z:
        a.append([1 / (1 + math.exp(-i))])
    pred = a
    a_list.append(a)
    #print("a" + str(n) + ": " + str(a))
    #print("Predicted output for instance: " + str(pred))
    #print("Expected output for instance: " + str(instance[1]))
    return pred, a_list, weights_list

def cost_function(layer_num, neuron_list, weights_list, trainingset, classes_list, lambda_value):
    J = 0
    for ins in trainingset:
        y = []
        if len(classes_list) == 1:
            y.append([ins[1]])
        elif isinstance(classes_list[0], int):
            for c in ins[1]:
                y.append([c])
        else:
            for c in classes_list:
                if ins[1] == c:
                    y.append([1.0])
                else:
                    y.append([0.0])
        output = forward_propagation(layer_num, neuron_list, weights_list, ins, len(classes_list))
        weights_list = output[2]
        cur_J = []
        for j in range(len(output[0])):
            cur_J.append([-y[j][0] * math.log(output[0][j][0]) - (1-y[j][0]) * math.log(1-output[0][j][0])])
        sum_J = 0
        for j in cur_J:
            sum_J += j[0]
        J += sum_J
        #print("Cost, J , associated with instance: " + str(sum_J))
    J /= len(trainingset)
    S = 0
    for layer in weights_list:
        for row in layer:
            new_row = row[1:]
            for r in new_row:
                S += r*r
    S *= (lambda_value / (2*len(trainingset)))
    #print("Final (regularized) cost, J, based on the complete training set: " + str(J+S))
    return J+S

def backpropagation(layer_num, neuron_list, weights_list, trainingset, classes_list, lambda_value):
    ini_J = cost_function(layer_num, neuron_list, weights_list, trainingset, classes_list, lambda_value)
    D_list = []
    for i in range(layer_num + 1):
        D_list.append([])
    for instance in trainingset:
        output, a_list, wl = forward_propagation(layer_num, neuron_list, weights_list, instance, len(classes_list))
        weights_list = wl
        y = []
        delta_list = []
        if len(classes_list) == 1:
            y.append([instance[1]])
        elif isinstance(classes_list[0], int):
            for c in instance[1]:
                y.append([c])
        else:
            for c in classes_list:
                if instance[1] == c:
                    y.append([1.0])
                else:
                    y.append([0.0])
        delta = np.array(output) - np.array(y)
        delta_list.insert(0, delta)
        #print("delta: " + str(delta))
        for num in range(layer_num):
            curweight = weights_list[-num-1]
            tweight = np.transpose(np.array(curweight))
            delta = np.dot(tweight, delta)
            for i in range(len(delta)):
                delta[i][0] = delta[i][0] * a_list[-num-2][i][0] * (1-a_list[-num-2][i][0])
            delta = delta[1:]
            delta_list.insert(0, delta)
            #print("delta: " + str(delta))
        for num in range(layer_num + 1):
            cur = np.array(D_list[-num-1])
            if len(cur) == 0:
                D_list[-num - 1] = np.dot(delta_list[-num-1], np.transpose(a_list[-num-2]))
            else:
                D_list[-num-1] = cur + np.dot(delta_list[-num-1], np.transpose(a_list[-num-2]))
        #for i in range(len(delta_list)):
            #print("Gradients of Theta" + str(len(delta_list) - i) + " based on training instance: " + str(np.dot(delta_list[-i-1], np.transpose(a_list[-i-2]))))
    for layer in range(layer_num+1):
        P = []
        for w in weights_list[-layer-1]:
            lst = []
            for e in w:
                lst.append(lambda_value * e)
            lst[0] = 0
            P.append(lst)
        D_list[-layer-1] = (1/len(trainingset)) * (np.array(D_list[-layer-1]) + np.array(P))
    #for i in range(len(D_list)):
        #print("Final regularized gradients of Theta" + str(i+1) + ": " + str(D_list[i]))
    for layer in range(layer_num+1):
        weights_list[-layer-1] = np.array(weights_list[-layer-1]) - 2 * np.array(D_list[-layer-1])
    cur_J = cost_function(layer_num, neuron_list, weights_list, trainingset, classes_list, lambda_value)
    if ini_J - cur_J < 0.001:
        return weights_list
    else:
        backpropagation(layer_num, neuron_list, weights_list, trainingset, classes_list, lambda_value)

def cal_performance(predictions, testset, classes, class_index):
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
            actual = testset[i][1]
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
        elif (tp + fp) == 0:
            precision = tn / (tn + fn)
            recall = tp / (tp + fn)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1 = 2 * ((precision * recall) / (precision + recall))
        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision
        total_F1 += F1
        #print("tp" + str(tp) + "tn" + str(tn) + "fp" + str(fp) + "fn" + str(fn))
    class_num = len(list(classes.keys()))
    total_accuracy = total_accuracy / class_num
    total_precision = total_precision / class_num
    total_recall = total_recall / class_num
    total_F1 = total_F1 / class_num
    return total_accuracy, total_precision, total_recall, total_F1

def cross_validation(examples, k, class_index,layer_num, neuron_list, lambda_value, num, header):
    classes = {}
    instances = []
    copyexamples = []
    for i in examples:
        copyexamples.append(i.copy())
    for i in range(len(copyexamples)):
        classname = copyexamples[i].pop(class_index)
    copyexamples = normalization(copyexamples, header, class_index)
    for i in range(len(examples)):
        curclass = examples[i][class_index]
        if curclass not in classes:
            classes[curclass] = [(copyexamples[i], curclass)]
        else:
            classes[curclass].append((copyexamples[i], curclass))
    folds = {}
    if num is not None:
        class_num = len(list(classes.keys()))
        new_classes = {}
        N = 0
        for i in classes:
            new_classes[i] = []
            for n in range(math.floor(num/class_num)):
                cur = None
                if len(classes[i]) == 0:
                    cur = random.choice(list(classes.values())[1])
                    N += 1
                    new_classes[i].append(cur)
                else:
                    cur = random.choice(classes[i])
                    N += 1
                    new_classes[i].append(cur)
                    classes[i].remove(cur)
        for i in range(num - N):
            curkey = random.choice(list(classes.keys()))
            cur = None
            if len(classes[curkey]) == 0:
                cur = random.choice(list(classes.values())[1])
                new_classes[cur[1]].append(cur)
            else:
                cur = random.choice(classes[curkey])
                classes[curkey].remove(cur)
                new_classes[curkey].append(cur)
        classes = new_classes
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
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    for i in folds:
        trainingset = []
        testset = []
        all_predictions = []  # store predictions for all test instances
        for j in folds:
            if i == j:
                testset.extend(folds[i])
            else:
                trainingset.extend(folds[i])
        final_weights_list = backpropagation(layer_num, neuron_list, None, trainingset, list(classes.keys()),
                                             lambda_value)
        J = cost_function(layer_num, neuron_list, final_weights_list, trainingset, list(classes.keys()), lambda_value)
        for j in testset:
            predictions = forward_propagation(layer_num, neuron_list, final_weights_list, j, len(list(classes.keys())))[
                0]
            pred = ""
            prob = 0
            for c in range(len(predictions)):
                if predictions[c][0] > prob:
                    prob = predictions[c][0]
                    pred = list(classes.keys())[c]
            all_predictions.append(pred)
        acc, pre, rec, F1 = cal_performance(all_predictions, testset, classes, class_index)
        total_accuracy += acc
        total_precision += pre
        total_recall += rec
        total_F1 += F1
    return total_accuracy / k, total_precision / k, total_recall / k, total_F1 / k, J

def run_nn(folds, k, class_index,layer_num, neuron_list, lambda_value, num, classes):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    for i in folds:
        trainingset = []
        testset = []
        all_predictions = []  # store predictions for all test instances
        for j in folds:
            if i == j:
                testset.extend(folds[i])
            else:
                trainingset.extend(folds[i])
        final_weights_list = backpropagation(layer_num, neuron_list, None, trainingset, list(classes.keys()),
                                             lambda_value)
        J = cost_function(layer_num, neuron_list, final_weights_list, trainingset, list(classes.keys()), lambda_value)
        for j in testset:
            predictions = forward_propagation(layer_num, neuron_list, final_weights_list, j, len(list(classes.keys())))[
                0]
            pred = ""
            prob = 0
            for c in range(len(predictions)):
                if predictions[c][0] > prob:
                    prob = predictions[c][0]
                    pred = list(classes.keys())[c]
            all_predictions.append(pred)
        acc, pre, rec, F1 = cal_performance(all_predictions, testset, classes, class_index)
        total_accuracy += acc
        total_precision += pre
        total_recall += rec
        total_F1 += F1
    return total_accuracy / k, total_precision / k, total_recall / k, total_F1 / k, J

def learning_curve(filename, examples, class_index, header):
    length = 0
    x = []
    J = []
    if filename == "titanic.csv":
        length = 887
    elif filename == "parkinsons.csv":
        length = 195
    elif filename == "cmc.data":
        length = 1473
    else:
        length = 435
    for i in range(10):
        num = math.floor((length)/20) * (i+1)
        if num > length:
            break
        x.append(num)
        accuracy = 0
        F1 = 0
        accuracy, precision, recall, F1, J1 = cross_validation(examples, 5, class_index,2, [8, 6], 0.6, num, header.copy())
        J.append(J1)
    plt.plot(x, J, label="")
    plt.xlabel("number of training samples")
    plt.ylabel("J")
    plt.errorbar(x, J, yerr=None, fmt='-o')
    plt.show()