/*
Author: Manohar Mukku
Date: 29.07.2018
Desc: To classify the test dataset on the trained parameter weights
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "mlp_classifier.h"

#define max(x, y) (x > y ? x : y)

void mat_mul_classify(double* a, double** b, double* result, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // matrix result of size 1 x p (array)
    // result = a * b
    int j, k;
    for (j = 0; j < p; j++) {
        result[j] = 0.0;
        for (k = 0; k < n; k++)
            result[j] += (a[k] * b[k][j]);
    }
}

void identity_classify(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = input[i]; // Identity function
}

void sigmoid_classify(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = 1.0 / (1.0 + exp(-input[i])); // Sigmoid function
}

void tan_h_classify(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = tanh(input[i]); // tanh function
}

void relu_classify(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = max(0.0, input[i]); // ReLU function
}

void softmax_classify(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
        sum += exp(input[i]);

    for (i = 0; i < n; i++) 
        output[i+1] = exp(input[i]) / sum; // Softmax function
}

void mlp_classifier(parameters* param, int* layer_sizes) {
    int n_layers = param->n_hidden + 2;

    // Create memory for arrays of inputs to the layers
    double** layer_inputs = (double**)calloc(n_layers, sizeof(double*));

    int i;
    for (i = 0; i < n_layers; i++)
        layer_inputs[i] = (double*)calloc(layer_sizes[i], sizeof(double));

    // Create memory for arrays of outputs from the layers
    double** layer_outputs = (double**)calloc(n_layers, sizeof(double*));

    for (i = 0; i < n_layers; i++)
        layer_outputs[i] = (double*)calloc(layer_sizes[i]+1, sizeof(double));


    // Classify the test dataset on the test samples
    int test_example;
    for (test_example = 0; test_example < param->test_sample_size; test_example++) {
        // Fill the input layer's input and output (both are equal) from data_test matrix for the given test example
        layer_outputs[0][0] = 1; // Bias term of input layer
        for (i = 0; i < param->feature_size-1; i++)
            layer_outputs[0][i+1] = layer_inputs[0][i] = param->data_test[test_example][i];

        // Perform forward propagation for each hidden layer
        // Calculate input and output of each hidden layer
        for (i = 1; i < n_layers-1; i++) {
            // Compute layer_inputs[i]
            mat_mul_classify(layer_outputs[i-1], weight[i-1], layer_inputs[i], layer_sizes[i-1]+1, layer_sizes[i]);

            // Compute layer_outputs[i]
            // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
            switch (param->hidden_activation_functions[i-1]) {
                case 1: // identity
                    identity_classify(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                    break;
                case 2: // sigmoid
                    sigmoid_classify(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                    break;
                case 3: // tanh
                    tan_h_classify(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                    break;
                case 4: // relu
                    relu_classify(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                    break;
                case 5: // softmax
                    softmax_classify(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                    break;
                default:
                    printf("Forward propagation: Invalid hidden activation function\n");
                    exit(0);
                    break;
            }
        }

        // Fill the output layers's input and output
        mat_mul_classify(layer_outputs[n_layers-2], weight[n_layers-2], layer_inputs[n_layers-1], layer_sizes[n_layers-2]+1, layer_sizes[n_layers-1]);

        // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
        switch (param->output_activation_function) {
            case 1: // identity
                identity_classify(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
                break;
            case 2: // sigmoid
                sigmoid_classify(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
                break;
            case 3: // tanh
                tan_h_classify(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
                break;
            case 4: // relu
                relu_classify(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
                break;
            case 5: // softmax
                softmax_classify(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
                break;
            default:
                printf("Forward propagation: Invalid hidden activation function\n");
                exit(0);
                break;
        }

        // Save the computed output into a output matrix
        // Final computed output is present in layer_outputs[n_layers-1] from index 1

    }

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers; i++)
        free(layer_outputs[i]);

    free(layer_outputs);

    for (i = 0; i < n_layers; i++)
        free(layer_inputs[i]);

    free(layer_inputs);
}