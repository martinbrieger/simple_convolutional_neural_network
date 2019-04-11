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
|Number of hidden-nodes:|200 is a sweet spot, more don't bring any improvements|
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

## Code
'''
dsd
'''

## Resources
- [Rashid Tariq, Make Your Own Neural Network](https://www.amazon.com/dp/1530826608/ref=cm_sw_em_r_mt_dp_U_AhERCbJ9PXK12)
- [Neural networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)