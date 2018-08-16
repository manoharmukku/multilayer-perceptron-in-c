# Implementation of Multi Layer Perceptron in C

Multi Layer perceptron (MLP) is an artificial neural network with one or more hidden layers between input and output layer. Refer to the following figure:

![MLP Network with one input layer, two hidden layers and an output layer](/figures/mlp-network.png)

Image from [Karim, 2016](https://dzone.com/articles/deep-learning-via-multilayer-perceptron-classifier). A multilayer perceptron with six input neurons, two hidden layers, and one output layer.

MLP's are fully connected (each hidden node is connected to each input node etc.). They use backpropagation as part of their learning phase. MLPs are widely used for pattern classification, recognition, prediction and approximation. Multi Layer Perceptron can solve problems which are not linearly separable ([Neuroph](http://neuroph.sourceforge.net/tutorials/MultiLayerPerceptron.html)).

## About this implementation:

This implementation of MLP was written using C and can perform multi-class classification. Each of the hidden layers and the output layer can run their own activation functions which can be specified during runtime. Supported activation functions are:

- identity ```f(x) = x```
- sigmoid ```f(x) = 1/(1 + e^-x)```
- tanh ```f(x) = tanh(x)```
- relu ```f(x) = max(0, x)```
- softmax ```f(x) = e^x / sum(e^x)```

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
~$ ./MLP 3 4,5,5 softmax,relu,tanh 1 sigmoid 0.01 10000 data/data_train.csv 1096 5 data/data_test.csv 275 5
```

Program parameters explanation:

> Argument 0: Executable file name _Ex:_ __./MLP__

> Argument 1: Number of hidden layers _Ex:_ __3__

> Argument 2: Number of units in each hidden layer from left to right separated by comma (no spaces in-between) _Ex:_ __4,5,5__

> Argument 3: Activation function of each hidden layer from left to right separated by comma (no spaces in-between) _Ex:_ __softmax,relu,tanh__

> Argument 4: Number of units in output layer (Specify 1 for binary classification and k for k-class multi-class classification) _Ex:_ __1__

> Argument 5: Output activation function _Ex:_ __sigmoid__

> Argument 6: Learning rate parameter _Ex:_ __0.01__

> Argument 7: Maximum number of iterations to run during training _Ex:_ __10000__

> Argument 8: Path of the csv file containing the train dataset _Ex:_ __data/data_train.csv__

> Argument 9: Number of rows in the train dataset (Number of samples) _Ex:_ __1096__

> Argument 10: Number of columns in the train dataset (Number of input features + 1 (output variable)). The output variable should always be in the last column _Ex:_ __5__

> Argument 11: Path of the csv file containing the test dataset _Ex:_ __data/data_test.csv__

> Argument 12: Number of rows in the test dataset (Number of samples) _Ex:_ __275__

> Argument 13: Number of columns in the test dataset (Number of input features + 1 (output variable)). The output variable should always be in the last column _Ex:_ __5__

## Dataset format:

1. The datasets should be in __.csv format__
1. All the __feature values__ should be __real__ (numerical)
1. There should _not_ be any header row
1. There should _not_ be any index column specifying the row number
1. The __output variable__ should always be in the __last column__
1. If __binary classification__, the __output variable__ can take values from __0 or 1 only__
1. If __multi-class classification__ with k-classes, the __output variable__ can take values from __1, 2, 3,..., k only__
1. Make sure to __specify the correct paths__ of the data files in the arguments

Example dataset used for training and testing: [Banknote authentication dataset from ULI ML Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

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
* https://www.geeksforgeeks.org/shuffle-a-given-array/
