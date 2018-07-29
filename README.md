# Implementation of Multi Layer Perceptron in C

Multi Layer perceptron (MLP) is an artificial neural network with one or more hidden layers between input and output layer. Refer to the following figure:

![MLP Network with one input layer, two hidden layers and an output layer](/figures/mlp-network.png)

Image from [Karim, 2016](https://dzone.com/articles/deep-learning-via-multilayer-perceptron-classifier). A multilayer perceptron with six input neurons, two hidden layers, and one output layer.

MLP's are fully connected (each hidden node is connected to each input node etc.). They use backpropagation as part of their learning phase. MLPs are widely used for pattern classification, recognition, prediction and approximation. Multi Layer Perceptron can solve problems which are not linearly separable ([Neuroph](http://neuroph.sourceforge.net/tutorials/MultiLayerPerceptron.html)).

## About this implementation:

This implementation of MLP was written using C and can perform multi-class classification. Each of the hidden layers and the output layer can run their own activation functions which can be specified during runtime. Supported activation functions are:

- Identity
- Sigmoid
- Tanh
- Relu
- Softmax

## How to run:

First, clone the project:

```
~$ git clone https://github.com/manoharmukku/multilayer-perceptron-in-c
```

Then, go to the cloned directory, and compile the project as below:

```
~$ make
```


Then run the program with your desired parameters as below:

```
~$ ./MLP 3 4,5,5 sigmoid,relu,tanh 3 softmax 0.01 10000 data_train.csv 1000 11
```

Program parameters explanation:

> Argument 0: Executable file name _Ex:_ __./MLP__

> Argument 1: Number of hidden layers _Ex:_ 3

> Argument 2: Size of each hidden layer separated by comma (no spaces in-between) _Ex:_ __4,5,5__

> Argument 3: Hidden activation functions separated by comma (no spaces in-between) _Ex:_ __sigmoid,relu,tanh__

> Argument 4: Number of units in output layer (Number of classes) _Ex:_ __3__

> Argument 5: Output activation function _Ex:_ __softmax__

> Argument 6: Learning rate parameter _Ex:_ __0.01__

> Argument 7: Maximum number of iterations _Ex:_ __10000__

> Argument 8: Name of the csv file containing the train dataset _Ex:_ __data_train.csv__

> Argument 9: Number of rows in the train dataset (Number of samples) _Ex:_ __1000__

> Argument 10: Number of columns in the train dataset (Number of features + 1(output variable)) _Ex:_ __11__

#### References:

* https://www.coursera.org/lecture/machine-learning/backpropagation-algorithm-1z9WW
* https://gist.github.com/amirmasoudabdol/f1efda29760b97f16e0e
* https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
* https://stackoverflow.com/questions/33058848/generate-a-random-double-between-1-and-1
* https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html
* http://dai.fmph.uniba.sk/courses/NN/haykin.neural-networks.3ed.2009.pdf
* https://jamesmccaffrey.wordpress.com/2017/06/23/two-ways-to-deal-with-the-derivative-of-the-relu-function/
* https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
* https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
