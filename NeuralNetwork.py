import numpy as np

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def one_hot(y, num_outputs):
    y_out = np.zeros((y.size, num_outputs))
    y_out[np.arange(y.size), y] = 1
    return y_out.T

class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_weights = np.random.rand(input_layer_size, hidden_layer_size).T - 0.5
        self.hidden_bias = np.random.rand(hidden_layer_size, 1) - 0.5
        self.output_weights = np.random.rand(hidden_layer_size, output_layer_size).T - 0.5
        self.output_bias = np.random.rand(output_layer_size, 1) - 0.5
        self.learning_rate = 0.05

    def set_weights_and_biases(self, hidden_weights, hidden_bias, output_weights, output_bias):
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias

    def train(self, inputs, expected):
        hidden_z = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_a = relu(hidden_z)

        output_z = np.dot(self.output_weights, hidden_a) + self.output_bias
        output_a = softmax(output_z)

        m = expected.size
        expected = one_hot(expected, self.output_layer_size)

        d_output_z = output_a - expected
        d_output_weights = 1/m*np.dot(d_output_z, hidden_a.T)
        d_output_bias = 1/m*np.sum(d_output_z)

        d_hidden_z = np.dot(self.output_weights.T, d_output_z)*relu_derivative(hidden_z)
        d_hidden_weights = 1/m*np.dot(d_hidden_z, inputs.T)
        d_hidden_bias = 1/m*np.sum(d_hidden_z)

        self.hidden_weights -= self.learning_rate*d_hidden_weights
        self.hidden_bias -= self.learning_rate*d_hidden_bias
        self.output_weights -= self.learning_rate*d_output_weights
        self.output_bias -= self.learning_rate*d_output_bias

    def predict(self, inputs):
        hidden_z = np.dot(self.hidden_weights, inputs) + self.hidden_bias
        hidden_a = relu(hidden_z)

        output_z = np.dot(self.output_weights, hidden_a) + self.output_bias
        output_a = softmax(output_z)

        return output_a

    def get_weights_and_biases():
        return self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias
