import math
import random
import numpy
from scipy.spatial import distance
from consts import NORMALIZE_PARAMETER

class Kohonen:

    def __init__(self, data, alpha, weight_rows, weight_columns):
        self.data = data
        self.alpha = alpha
        self.weights = self.get_weights(weight_rows, weight_columns)
        self.error = []

    def get_weights(self, rows, columns):
        return [[random.uniform(0.01, 0.99) for j in range(columns)] for i in range(rows)]

    # Function here computes the winning vector by Euclidean distance
    def get_neuron_winner_index(self, vector):
        distance_list = numpy.zeros(len(self.weights))

        for row in range(len(self.weights)):
            for j in range(len(vector)):
                distance_list[row] += math.pow((vector[j] - self.weights[row][j]), 2)

        for i in range(len(distance_list)):
            distance_list[i] = math.sqrt(distance_list[i])

        return self.get_index_of_min(distance_list)

    def get_index_of_min(self, array):
        min = array[0]
        index = 0
        for i in range(len(array)):
            if array[i] < min:
                min = array[i]
                index = i
        return index

    def get_normalize_outputs(self, data):
        output_values = []
        for window_index in range(len(data) - self.window_size + 1):
            current_window = data[window_index: window_index + self.window_size]
            output_values.append(self.get_normalize_output(self.get_distances_relative_to_neurons(current_window)))

        return output_values

    def get_normalize_output(self, distances_by_vector):
        normalize_output_by_vector = []
        winner_neuron_output = min(distances_by_vector)
        for i in range(len(distances_by_vector)):
            normalize_output_by_vector.append(self.get_normalize_value(distances_by_vector[i], winner_neuron_output))

        return normalize_output_by_vector

    def get_normalize_value(self, neuron_output, winner_neuron_output, q=NORMALIZE_PARAMETER):
        return math.exp(-(math.pow(abs(neuron_output - winner_neuron_output), 2) / math.pow(q, 2)))

    def calculate_error(self):
        error = 0
        error_dist = []
        for inp in self.data:
            for i in self.weights:
                error_dist.append(distance.euclidean(i, inp))
            error += min(error_dist) ** 2
            error_dist.clear()
        self.error.append(error / len(self.data))

    # Function here updates the winning vector
    def update_weights(self, sample, winner_neuron_index):
        for i in range(len(self.weights)):
            self.weights[winner_neuron_index][i] = self.weights[winner_neuron_index][i] + self.alpha * (
                    sample[i] - self.weights[winner_neuron_index][i])

    def train(self, epochs):
        for epoch_index in range(epochs):
            for row_index in range(len(self.data)):
                current_vector = self.data[row_index]
                winner_neuron_index = self.get_neuron_winner_index(current_vector)
                self.update_weights(current_vector, winner_neuron_index)
            self.calculate_error()
            print('epoch: ' + str(epoch_index) + ' - ' + str(self.error[epoch_index]))
        return self.error

    def test(self, data):
        target = []
        for row_index in range(len(data)):
            current_vector = data[row_index]
            winner_neuron_index = self.get_neuron_winner_index(current_vector)
            target.append(winner_neuron_index)
        return target
