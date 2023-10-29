# python main.py train/train7.dat test/test7.dat 0 1 0.9 1000
# python main.py train/train7.dat test/test7.dat 1 2 0.9 1000
import sys
import pandas as pd
import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

class Node:
    def __init__(self, num_inputs, node_num) -> None:
        self.value = None
        self.weights = np.zeros(num_inputs + 1)
        self.inputs = None
        self.delta = None
        self.node_num = node_num

    def calc_value(self, inputs):
        self.inputs = np.array(inputs)
        # Add bias
        self.inputs = np.append(self.inputs, 1)
        self.value = sigmoid(np.dot(self.inputs, self.weights))
        return self.value

    def backprop_output(self, actual):
        self.delta = self.value*(1 - self.value)*(actual - self.value)
        self.weights += learning_rate * self.delta * self.inputs
        
    def backprop_hidden(self, forward_layer):
        forward_weights = forward_layer.get_forward_weights(self.node_num)
        forward_deltas = forward_layer.get_forward_deltas()
        self.delta = self.value*(1 - self.value)*np.dot(forward_weights, forward_deltas)
        self.weights += learning_rate * self.delta * self.inputs

class Layer:
    def __init__(self, num_inputs, num_hidden_nodes) -> None:
        self.num_inputs = num_inputs
        self.num_hidden_nodes = num_hidden_nodes
        self.nodes = [Node(num_inputs, node_num) for node_num in range(num_hidden_nodes)]
        self.output = None

    def forward(self, input: list[int]):
        self.output = [node.calc_value(input) for node in self.nodes]
        return self.output

    def get_forward_weights(self, node_num):
        return [node.weights[node_num] for node in self.nodes]

    def get_forward_deltas(self):
        return [node.delta for node in self.nodes]
    
    def backprop_output(self, actual):
        for node in self.nodes:
            node.backprop_output(actual)
    
    def backprop_hidden(self, forward_layer):
        for node in self.nodes:
            node.backprop_hidden(forward_layer)


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    num_hidden_layers = int(sys.argv[3])
    num_hidden_nodes = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    num_iterations = int(sys.argv[6])
    
    # Read data
    train_data = read_data(train_file)
    test_data = read_data(test_file)
    # Get num_inputs from train_data
    num_inputs = train_data.shape[1] - 1
    # Turn to numpy array

    if num_hidden_layers == 0:
        # Initialize layers
        output_layer = Layer(num_inputs, 1)
        # # Forward propagation
        num_rows = train_data.shape[0]
        for i in range(num_iterations):
            index = i % num_rows  # Wrap around using modulo
            row = train_data.iloc[index]  # Use iloc to get the row by integer-based index
    
            input_data = row[:-1].to_numpy()
            actual = row[-1]
    
            # Forward propagation
            output = output_layer.forward(input_data)
    
            # Backward propagation
            output_layer.backprop_output(actual)
    
            print(f"  Iteration: {i+1}")
            print(f"  Input: {input_data}")
            print(f"  Output: {output[0]:.4f}")
            print(f"  Weights: {[node.weights for node in output_layer.nodes]}")

    else:
        # Initialize layers
        input_layer = Layer(num_inputs, num_hidden_nodes)
        hidden_layers = [Layer(num_hidden_nodes, num_hidden_nodes) for _ in range(num_hidden_layers - 1)]
        # hidden_layers = [Layer(num_hidden_nodes, num_hidden_nodes) for _ in range(num_hidden_layers + 1)]
        # hidden_layers = [Layer(num_hidden_nodes, num_hidden_nodes) for _ in range(num_hidden_layers)]
        output_layer = Layer(num_hidden_nodes, 1)

        num_rows = train_data.shape[0]
        for i in range(num_iterations):
        # for i in range(8):
            index = i % num_rows  # Wrap around using modulo
            row = train_data.iloc[index]  # Use iloc to get the row by integer-based index
            input_data = row[:-1].to_numpy()
            actual = row[-1]
            # Forward propagation
            # hidden_layer_outputs = input_layer.forward(input_data)
            # for hidden_layer in hidden_layers:
            #     hidden_layer_outputs = hidden_layer.forward(hidden_layer_outputs)
            # output = output_layer.forward(hidden_layer_outputs)

            hidden_layer_outputs = input_layer.forward(input_data)
            output = output_layer.forward(hidden_layer_outputs)

            # Backward propagation
            output_layer.backprop_output(actual)

            # prev_layer = output_layer
            # for hidden_layer in reversed(hidden_layers):
            #     hidden_layer.backprop_hidden(prev_layer)
            #     prev_layer = hidden_layer
            prev_layer = output_layer
            input_layer.backprop_hidden(prev_layer)

            print(f"  Iteration: {i+1}")
            print(f"  Input: {input_data}")
            print(f"  Output: {output[0]:.4f}")
            print(f"  Weights: {[node.weights for node in output_layer.nodes]}")

        # Print forward and backward propagation
        print(input)
        print(hidden_layer_outputs)
        print(output)
