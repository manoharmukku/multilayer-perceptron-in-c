/*
Author: Manohar Mukku
Date:   13.07.2018
GitHub: https://www.github.com/manoharmukku
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
Hidden Layer sizes: # of hidden layers, Size of each hidden layer separated by space (int)
Hidden activation function: identity, sigmoid, tanh, relu (char*)
Alpha (L2 Regularization parameter value): Float value (float)
Maximum number of iterations: > 0 (int)
Momentum for gradient descent update: Integer value (int)
Number of units in output layer: # of units in output layer (int)
Output activation: identity, sigmoid, tanh, relu (char*)
*/

void MLPClassifier(int n_hidden, int* hidden_layer_size, int hidden_activation_function, float regularization_parameter, int n_iterations_max, 
	int momentum, int output_layer_size, int output_activation_function) {
	
}

int main(int argc, char** argv) {
	// Sanity check of command line arguments
	if (argc < 1) {
		printf("Usage: 'No. of hidden layers' 'Size of each hidden layer separated by space' 'Hidden activation' 'Alpha' \n");
		exit(-1);
	}

	// Number of hidden layers
	int n_hidden = atoi(argv[1]);
	// Sanity check of number of hidden layers
	if (n_hidden < 0) {
		printf("Error: Number of hidden layers should be >= 0\n");
		exit(-1);
	}

	// Size of each hidden layer
	int* hidden_layer_size = (int*)malloc(n_hidden * sizeof(int));
	int i;
	for (i = 0; i < n_hidden; i++) {
		hidden_layer_size[i] = atoi(argv[i+2]);

		// Sanity check of size of hidden layer
		if (hidden_layer_size[i] <= 0) {
			printf("Error: Hidden layer sizes should be positive\n");
			exit(-1);
		}
	}

	// Hidden activation function
	int hidden_activation_function;
	switch (argv[n_hidden + 2]) {
		case "identity":
			hidden_activation_function = 1;
			break;
		case "sigmoid":
			hidden_activation_function = 2;
			break;
		case "tanh":
			hidden_activation_function = 3;
			break;
		case "relu":
			hidden_activation_function = 4;
			break;
		default:
			printf("Error: Invalid value for hidden activation function\n");
			printf("Input either identity or sigmoid or tanh or relu for hidden activation function\n");
			exit(-1);
			break;
	}

	// L2 Regularization parameter
	float regularization_parameter = atoi(argv[n_hidden + 3]);

	// Max. number of iterations
	int n_iterations_max = atoi(argv[n_hidden + 4]);
	if (n_iterations_max <= 0) {
		printf("Max. number of iterations value should be positive\n");
		exit(-1);
	}

	// Momentum
	int momentum = atoi(argv[n_hidden + 5]);

	// Output layer size
	int output_layer_size = atoi(argv[n_hidden + 6]);
	if (output_layer_size <= 0) {
		printf("Output layer size should be positive\n");
		exit(-1);
	}

	// Output activation function
	int output_activation_function;
	switch (argv[n_hidden + 7]) {
		case "identity":
			output_activation_function = 1;
			break;
		case "sigmoid":
			output_activation_function = 2;
			break;
		case "tanh":
			output_activation_function = 3;
			break;
		case "relu":
			output_activation_function = 4;
			break;
		default:
			printf("Error: Invalid value for output activation function\n");
			printf("Input either identity or sigmoid or tanh or relu for output activation function\n");
			exit(-1);
			break;
	}

	return 0;
}