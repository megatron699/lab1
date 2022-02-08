# -*- coding: utf-8 -*-

import math

from Neuron import Neuron
from NeuronLayer import NeuronLayer
from consts import EPOCHS


class NeuralNetwork:

    def __init__(self, topology):
        self.topology = topology
        self.layers = []
        self.create_layers()

    def create_layers(self):
        self._create_input_layer()
        self._create_hidden_layers()
        self._create_output_layers()

    def _create_input_layer(self):
        input_neurons = []
        for i in range(self.topology.get_input_count()):
            neuron = Neuron(1, "Input")
            input_neurons.append(neuron)
        layer = NeuronLayer(input_neurons)
        self.layers.append(layer)

    def _create_hidden_layers(self):
        for hidden_layer_count in self.topology.get_hidden_layer():
            hidden_neurons = []
            last_layer_count = len(self.layers[-1].neurons)
            for i in range(hidden_layer_count):
                neuron = Neuron(last_layer_count)
                hidden_neurons.append(neuron)
            hidden_layer = NeuronLayer(hidden_neurons)
            self.layers.append(hidden_layer)

    def _create_output_layers(self):
        output_neurons = []
        last_layer_count = len(self.layers[-1].neurons)
        for i in range(self.topology.get_output_count()):
            neuron = Neuron(last_layer_count, "Output")
            output_neurons.append(neuron)
        output_layer = NeuronLayer(output_neurons)
        self.layers.append(output_layer)

    def feed_forward(self, inputs):
        # Инпуты
        for i in range(len(inputs)):
            self.layers[0].neurons[i].feed_forward([inputs[i]])

        # Скрытые и выходной
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer_signal = self.layers[i - 1].get_signals()
            for neuron in layer.neurons:
                neuron.feed_forward(prev_layer_signal)
        return self.layers[-1].neurons

    # Обратное распространение ошибки
    def _back_propagation(self, expected_values, inputs, alpha):
        differences = []
        for index in range(len(self.layers[-1].neurons)):
            actual_layer = self.feed_forward(inputs)[index]
            actual = actual_layer.output
            expected = expected_values[index]
            difference = actual - expected
            differences.append(difference)

            # Нейроны последнего слоя учатся по difference
            self.layers[-1].neurons[index].learn(difference, alpha)

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i + 1]
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                for k in range(len(prev_layer.neurons)):
                    prev_neuron = prev_layer.neurons[k]
                    error = prev_neuron.weights[j] * prev_neuron.delta
                    # Нейроны скрытого слоя по error
                    neuron.learn(error, alpha)
        return sum([(difference ** 2) / len(self.layers[-1].neurons) for difference in differences])

    def learn(self, dataset, alpha):
        loss_arr = []
        for i in range(EPOCHS):
            error = 0
            for j in range(len(dataset['learn']['data'])):
                error += self._back_propagation(dataset['learn']['target'][j], dataset['learn']['data'][j], alpha)

            temp = math.sqrt(error / (EPOCHS - 1))
            print("{} - {}".format(i + 1, temp))
            loss_arr.append(temp)
            if temp < 0.001:
                print("learning ended on {} epoch".format(i + 1))
                break
        return temp, loss_arr

    def test(self, dataset):

        results = []
        answers = dataset['test']['target']
        for data in dataset['test']['data']:
            neurons = self.feed_forward(data)
            neurons_outputs = [neuron.output for neuron in neurons]
            max_neuron_output = max(neurons_outputs)
            neurons_outputs = [1 if max_neuron_output == neuron_output else 0 for neuron_output in neurons_outputs]
            results.append(neurons_outputs)

        total = len(answers)
        good = 0
        bad = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(total):
            if answers[i] == results[i]:
                good += 1
            else:
                print(answers[i], results[i])
                bad += 1
            if answers[i][0] == 1:
                count1 += 1
            if answers[i][0] == 1:
                count2 += 1
            if answers[i][0] == 1:
                count3 += 1

        print("Good = {}%, {}/{}\n"
              "Bad = {}%, {}/{}".format(good / float(total) * 100, good, total, bad / float(total) * 100, bad, total))

        return bad / float(total)

    def __str__(self):
        return "Topology = {}\n" \
               "Layers = {}\n" \
               "\n".format(self.topology,
                           [str(layer) for layer in self.layers])
