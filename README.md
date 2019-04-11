# Building a simple convolutional neural network with Python

This is my first neural network - built with Python, Scipy and Numpy. I used Jupyter Lab as IPython environment. Additionally regular .py-files are provided (see resources section).

## Goals
- Build a simple neural network that can classify handwritten digits
- Learn how neural networks work und understand the math behind them
- Improve Python skills
- Have fun playing around

## Specification
- Type: simple 3-layer [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- Train and test-data: [MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- Number of input-nodes: fixed 783 (one for each pixel of the handwritten digit)
- Number of output-nodes: fixed 10 (for digits from 0-9)
- Activation function: sigmoid

## Result: 
With the following parameters I'm able to get **~97% correct classifications or an error rate of ~3%**

|Setting||
|-|-|
|Number of hidden-nodes:|200 is the maximum that improves the network|
|Learning rate:|I tested values from 0.01 to 0.3 and found that 0.1 is roughly the optimum|
|Epochs:|5 seems like the optimum, higher values lead to overfitting|

## Index of files
|File||
|-|-|
|1_neural_network.ipynb|neural network class definition|
|2_neural_network+mnist_data.ipynb|+ training data|
|3_neural_network+mnist_data+scorecard.ipynb|+ scorecard|
|4_neural_network+full_dataset.ipynb|+ full database **(runs)**|
|5_neural_network+epochs.ipynb|+ epochs **(final version, runs)**|
|5_neural_network_final.py|↳ final .py version|

## Code
```python
import numpy
import scipy.special # for sigmoid function
import matplotlib.pyplot

class neuralNetwork:
    
    # 0. initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set the number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        
        # learning rate
        self.lr = learningrate
        
        # activation funktion (sigmoid function)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # 1. train the neural network
    def train(self, inputs_list, targets_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (taget - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    # 2. query the neural network
    def query(self, inputs_list):
        # converts input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learnig rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
```
```python
# load the mnist training data csv file into a list
training_data_file = open("mnist_datasets/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, exept the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load the mnist test data csv file into a list
test_data_file = open("mnist_datasets/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # if network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # if network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
```

## Resources
- [Rashid Tariq, Make Your Own Neural Network](https://www.amazon.com/dp/1530826608/ref=cm_sw_em_r_mt_dp_U_AhERCbJ9PXK12)
- [Neural networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)