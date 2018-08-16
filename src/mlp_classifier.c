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

    // Create memory to store final outputs
    double** final_output = (double**)calloc(param->test_sample_size, sizeof(double*));
    for (i = 0; i < param->test_sample_size; i++)
        final_output[i] = (double*)calloc(param->output_layer_size, sizeof(double));


    // Classify the test dataset on the test samples
    int test_example;
    for (test_example = 0; test_example < param->test_sample_size; test_example++) {
        printf("Classifying test example %d of %d\r", test_example+1, param->test_sample_size);
        // Fill the input layer's input and output (both are equal) from data_test matrix for the given test example
        layer_outputs[0][0] = 1; // Bias term of input layer
        for (i = 0; i < param->feature_size-1; i++)
            layer_outputs[0][i+1] = layer_inputs[0][i] = param->data_test[test_example][i];

        // Perform forward propagation for each hidden layer
        // Calculate input and output of each hidden layer
        for (i = 1; i < n_layers-1; i++) {
            // Compute layer_inputs[i]
            mat_mul_classify(layer_outputs[i-1], param->weight[i-1], layer_inputs[i], layer_sizes[i-1]+1, layer_sizes[i]);

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
        mat_mul_classify(layer_outputs[n_layers-2], param->weight[n_layers-2], layer_inputs[n_layers-1], layer_sizes[n_layers-2]+1, layer_sizes[n_layers-1]);

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
        for (i = 0; i < param->output_layer_size; i++)
            final_output[test_example][i] = layer_outputs[n_layers-1][i+1];
    }

    // Find the output class for each test example
    if (param->output_layer_size == 1) { // Binary classification
        for (test_example = 0; test_example < param->test_sample_size; test_example++) {
            if (final_output[test_example][0] < 0.5)
                final_output[test_example][0] = 0;
            else
                final_output[test_example][0] = 1;
        }
    }
    else { // Multi-class classification
        for (test_example = 0; test_example < param->test_sample_size; test_example++) {
            double max = -1;
            int max_class;
            for (i = 0; i < param->output_layer_size; i++) {
                if (final_output[test_example][i] > max) {
                    max = final_output[test_example][i];
                    max_class = i+1;
                }
            }
            final_output[test_example][0] = max_class;
        }
    }

    // Calculate the confusion matrix
    if (param->output_layer_size == 1) { // Binary classification
        int true_positive = 0, true_negative = 0, false_positive = 0, false_negative = 0;
        for (test_example = 0; test_example < param->test_sample_size; test_example++) {
            if (final_output[test_example][0] == 0) {
                if (param->data_test[test_example][param->feature_size-1] == 0)
                    ++true_negative;
                else
                    ++false_positive;
            }
            else {
                if (param->data_test[test_example][param->feature_size-1] == 1)
                    ++true_positive;
                else
                    ++false_negative;
            }
        }

        // Find the accuracy
        double accuracy = (double)(true_positive + true_negative) / param->test_sample_size;

        // Print confusion matrix
        printf("\n\nConfusion matrix\n");
        printf("-----------------\n\n");

        printf("\t    |predicted 0\t predicted 1\n");
        printf("--------------------------------------------\n");
        printf("Actual 0    |%d\t\t%d\n\n", true_negative, false_positive);
        printf("Actual 1    |%d\t\t%d\n\n", false_negative, true_positive);

        // Print the accuracy
        printf("\nAccuracy: %.2lf\n\n", accuracy * 100);
    }
    else { // Multi-class classification
        int** confusion_matrix = (int**)calloc(param->output_layer_size, sizeof(int*));
        for (i = 0; i < param->output_layer_size; i++)
            confusion_matrix[i] = (int*)calloc(param->output_layer_size, sizeof(int));

        // Fill the confusion matrix
        int actual_class, predicted_class;
        for (test_example = 0; test_example < param->test_sample_size; test_example++) {
            actual_class = param->data_test[test_example][param->feature_size-1] - 1;
            predicted_class = final_output[test_example][0] - 1;

            ++confusion_matrix[actual_class][predicted_class];
        }

        // Print the confusion matrix
        printf("\t");
        for (predicted_class = 1; predicted_class <= param->output_layer_size; predicted_class++)
            printf("Predicted %d  ", predicted_class);
        printf("\n---------------------------------------------------------------------------\n");

        for (actual_class = 0; actual_class < param->output_layer_size; actual_class++) {
            printf("Actual %d | ", actual_class+1);
            for (predicted_class = 0; predicted_class < param->output_layer_size; predicted_class++)
                printf("%d\t", confusion_matrix[actual_class][predicted_class]);
            printf("\n");
        }

        // Find the accuracy
        double accuracy = 0.0;
        for (i = 0; i < param->output_layer_size; i++)
            accuracy += confusion_matrix[i][i];
        accuracy /= param->test_sample_size;

        // Print the accuracy
        printf("\nAccuracy: %.2lf\n\n", accuracy * 100);

        // Free the memory allocated in heap
        for (i = 0; i < param->output_layer_size; i++)
            free(confusion_matrix[i]);
        free(confusion_matrix);
    }


    // Write the final output into a csv file
    char* output_file_name = "data/data_test_output.csv";
    write_csv(output_file_name, param->test_sample_size, param->output_layer_size, final_output);

    // Free the memory allocated in Heap
    for (i = 0; i < param->test_sample_size; i++)
        free(final_output[i]);

    free(final_output);

    for (i = 0; i < n_layers; i++)
        free(layer_outputs[i]);

    free(layer_outputs);

    for (i = 0; i < n_layers; i++)
        free(layer_inputs[i]);

    free(layer_inputs);
}