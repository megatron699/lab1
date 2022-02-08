# -*- coding: utf-8 -*-
import numpy as np

from NeuralNetwork import NeuralNetwork
from Topology import Topology
from consts import *
from Kohonen import *
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas
import consts


def get_data_from_framework():
    data = datasets.load_iris()

    data_list = {'minmax': {
        'min': numpy.amin(data['data'], axis=0).tolist(),
        'max': numpy.amax(data['data'], axis=0).tolist()
    },
        'data': data['data'].tolist(),

    }

    data_list['target'] = [[0 for i in range(NN_LAYERS[-1])] for j in range(len(data_list['data']))]

    for i in range(len(data_list['target'])):
        data_list['target'][i][data['target'][i]] = 1

    # print(data['data'])
    return data_list


def normalize(data):
    # x - xmin / xmax - xmin
    for i in range(len(data['data'][0])):
        current_min = data['minmax']['min'][i]
        current_max = data['minmax']['max'][i]
        delta = current_max - current_min
        for j in range(len(data['data'])):
            data['data'][j][i] = (data['data'][j][i] - current_min) / delta
    return data


def divide_data(data, learn_percentage):
    last_learn_item = int(len(data['data']) * learn_percentage)

    from sklearn.utils import shuffle
    data['data'], data['target'] = shuffle(data['data'], data['target'], random_state=RANDOM_SEED)

    data_list = {
        'learn': {
            'data': data['data'][:last_learn_item],
            'target': data['target'][:last_learn_item],
        },
        'test': {
            'data': data['data'][last_learn_item:],
            'target': data['target'][last_learn_item:],
        },
    }

    return data_list


def get_target_classes(target_indexes):
    target_classes = []
    for i in range(len(target_indexes)):
        tmp = [0, 0, 0]
        tmp[target_indexes[i]] = 1
        target_classes.append(tmp)
    return target_classes


def kohonen_out_to_ffinput(kohonen_out, data_train):
    target = [(x[1]) for x in data_train]
    target = np.asarray(target).reshape(len(target), 1)
    ff_input = np.concatenate((kohonen_out, target), axis=1)
    ff_input = [(x[:NN_LAYERS[0]][np.newaxis, :], int(x[NN_LAYERS[0]])) for x in ff_input]
    return ff_input

def train_and_test(learn_percentage=LEARN_PERCENTAGE, alpha_kohonen=ALPHA_KOHONEN, alpha=ALPHA, kohonen_layer_count=KOHONEN_LAYER_COUNT,
                   hidden_layer_count=HIDDEN_LAYER_COUNT):
    data = get_data_from_framework()
    data = normalize(data)
    kohonen_layer_network = Kohonen(data['data'], alpha_kohonen, kohonen_layer_count, len(data['data'][0]))
    error = kohonen_layer_network.train(EPOCHS_KOHONEN)
    target = kohonen_layer_network.test(data['data'])
    target_classes = get_target_classes(target)

    data['target'] = target_classes
    data = divide_data(data, learn_percentage)
    nn_layers = [13, hidden_layer_count, kohonen_layer_count]
    topology = Topology(nn_layers)
    nn = NeuralNetwork(topology)

    learn_error, _ = nn.learn(data, alpha)
    test_error = nn.test(data)
    return learn_error, test_error

def test_alpha():
    alpha_min = 0.05
    alpha_max = 0.55
    step = 0.05

    error_arr = []
    alpha_arr = np.arange(alpha_min, alpha_max, step)
   # orig = consts.ALPHA_KOHONEN
    for a in alpha_arr:
    #    consts.ALPHA_KOHONEN = a
        test_error = train_and_test(alpha=a)
        print(learn_error)
        print(test_error)
        error_arr.append(test_error)
   # consts.ALPHA_KOHONEN = orig
    return alpha_arr, error_arr
def test_alpha_kohonen():
    alpha_min = 0.05
    alpha_max = 0.55
    step = 0.05

    error_arr = []
    alpha_arr = np.arange(alpha_min, alpha_max, step)
   # orig = consts.ALPHA_KOHONEN
    for a in alpha_arr:
    #    consts.ALPHA_KOHONEN = a
        _, test_error = train_and_test(alpha_kohonen=a)
        print(learn_error)
        print(test_error)
        error_arr.append(test_error)
   # consts.ALPHA_KOHONEN = orig
    return alpha_arr, error_arr

def test_kohonen_neurons_count():
    kohonen_neurons_count_min = 1
    kohonen_neurons_count_max = 10
    step = 1
    error_arr = []
    neurons_count_arr = np.arange(kohonen_neurons_count_min, kohonen_neurons_count_max, step)
    # orig = consts.KOHONEN_LAYER_COUNT
    for a in neurons_count_arr:
        # consts.KOHONEN_LAYER_COUNT = a
        learn_error, test_error = train_and_test(kohonen_layer_count=a)
        print(learn_error)
        print(test_error)
        error_arr.append(test_error)
    # consts.KOHONEN_LAYER_COUNT = orig
    return neurons_count_arr, error_arr

def test_hidden_neurons_count():
    hidden_neurons_min = 1
    hidden_neurons_max = 10
    step = 1
    error_arr = []
    learn_error_arr = []
    hidden_neurons_arr = np.arange(hidden_neurons_min, hidden_neurons_max, step)

   # orig = consts.HIDDEN_LAYER_COUNT
    for a in hidden_neurons_arr:
      #  consts.HIDDEN_LAYER_COUNT = a
        learn_error, test_error = train_and_test(hidden_layer_count=a)
        print(learn_error)
        print(test_error)
        learn_error_arr.append(learn_error)
        error_arr.append(test_error)
   # consts.HIDDEN_LAYER_COUNT = orig
    return hidden_neurons_arr, error_arr, learn_error_arr

def test_learn_size():
    learn_size_min = 0.4
    learn_size_max = 0.9
    step = 0.05
    error_arr = []
    learns_size_arr = np.arange(learn_size_min, learn_size_max, step)

    #orig = consts.LEARN_PERCENTAGE
    for a in learns_size_arr:
        #consts.LEARN_PERCENTAGE = a
        test_error = train_and_test(learn_percentage=a)
        print(learn_error)
        print(test_error)
        error_arr.append(test_error)
    #consts.LEARN_PERCENTAGE = orig
    return learns_size_arr, error_arr

def test_epochs_count():
    epochs_min = 10
    epochs_max = 100
    step = 5
    error_arr = []
    epochs_arr = np.arange(epochs_min, epochs_max, step)

  #  orig = consts.EPOCHS
    for a in epochs_arr:
       # consts.EPOCHS = a
        test_error = train_and_test()
        print(learn_error)
        print(test_error)
        error_arr.append(test_error)
    #consts.EPOCHS = orig
    return epochs_arr, error_arr


if __name__ == "__main__":
    data = get_data_from_framework()
    data = normalize(data)

    # Kohonen
    print('Kohonen layer')
    kohonen_layer_network = Kohonen(data['data'], ALPHA_KOHONEN, KOHONEN_LAYER_COUNT, len(data['data'][0]))
    error = kohonen_layer_network.train(EPOCHS_KOHONEN)
    target = kohonen_layer_network.test(data['data'])
    target_classes = get_target_classes(target)
    #plt.plot(error)
    #plt.show()

    # MLP
    print('MLP')
    data['target'] = target_classes
    data = divide_data(data, LEARN_PERCENTAGE)
    topology = Topology(NN_LAYERS)
    nn = NeuralNetwork(topology)
    learn_error, loss_arr = nn.learn(data, ALPHA)
    test_error = nn.test(data)
    print(learn_error)
    print(test_error)
    # plt.plot(loss_arr)
    # plt.show()

    # alpha_arr, error_arr = test_alpha()
    # plt.xlabel('ALPHA')
    # plt.ylabel('error')
    # plt.plot(alpha_arr, error_arr)
    # plt.show()

    # alpha_arr, error_arr = test_alpha_kohonen()
    # plt.xlabel('ALPHA_KOHONEN')
    # plt.ylabel('error')
    # plt.plot(alpha_arr, error_arr)
    # plt.show()

    # kohonen_neurons_count, error_arr = test_kohonen_neurons_count()
    # plt.xlabel('Kohonen neurons count')
    # plt.ylabel('error')
    # plt.plot(kohonen_neurons_count, error_arr)
    # plt.show()

    # learn_size, error_arr = test_learn_size()
    # plt.xlabel('learn_size')
    # plt.ylabel('error')
    # plt.plot(learn_size, error_arr)
    # plt.show()
    #
    # hidden_count, error_arr, learn_error_arr = test_hidden_neurons_count()
    # plt.xlabel('hidden_count')
    # plt.ylabel('error')
    # plt.plot(hidden_count, error_arr)
    # plt.show()

    epochs, error_arr = test_epochs_count()
    plt.xlabel('epochs_count')
    plt.ylabel('error')
    plt.plot(epochs, error_arr)
    plt.show()
