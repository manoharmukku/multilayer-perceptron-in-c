/*
Author: Manohar Mukku
Date: 23.07.2018
Desc: Backpropagation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "back_propagation.h"

void d_identity(int layer_size, double* layer_input, double* layer_output, double* layer_derivative) {
    int i;
    for (i = 0; i < layer_size; i++)
        layer_derivative[i] = 1;
}

void d_sigmoid(int layer_size, double* layer_input, double* layer_output, double* layer_derivative) {
    int i;
    for (i = 0; i < layer_size; i++)
        layer_derivative[i] = layer_output[i+1] * (1.0 - layer_output[i+1]);
}

void d_tanh(int layer_size, double* layer_input, double* layer_output, double* layer_derivative) {
    int i;
    for (i = 0; i < layer_size; i++)
        layer_derivative[i] = 1.0 - layer_output[i+1] * layer_output[i+1];
}

void d_relu(int layer_size, double* layer_input, double* layer_output, double* layer_derivative) {
    int i;
    for (i = 0; i < layer_size; i++) {
        if (layer_input[i] > 0)
            layer_derivative[i] = 1;
        else if (layer_input[i] < 0)
            layer_derivative[i] = 0;
        else // derivative does not exist
            layer_derivative[i] = 0.5; // giving arbitrary value
    }
}

void d_softmax(int layer_size, double* layer_input, double* layer_output, double* layer_derivative) {
    int i;
    for (i = 0; i < layer_size; i++)
        layer_derivative[i] = layer_output[i+1] * (1.0 - layer_output[i+1]);
}

void calculate_local_gradient(parameters* param, int layer_no, int n_layers, int* layer_sizes, double** layer_inputs, double** layer_outputs,
    double* expected_output, double** local_gradient) {
    // Create memory for derivatives
    double** layer_derivatives = (double**)calloc(n_layers, sizeof(double*));

    int i;
    for (i = 0; i < n_layers; i++)
        layer_derivatives[i] = (double*)calloc(layer_sizes[i], sizeof(double));

    // If output layer
    if (layer_no == n_layers-1) {
        // Error produced at the output layer
        double* error_output = (double*)calloc(param->output_layer_size, sizeof(double));

        for (i = 0; i < param->output_layer_size; i++)
            error_output[i] = expected_output[i] - layer_outputs[layer_no][i+1];

        // Calculate the layer derivatives
        // Calculate the local gradients
        switch(param->output_activation_function) {
            case 1: // identity
                d_identity(param->output_layer_size, layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < param->output_layer_size; i++)
                    local_gradient[layer_no][i] = error_output[i] * layer_derivatives[layer_no][i];

                break;
            case 2: // sigmoid
                d_sigmoid(param->output_layer_size, layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < param->output_layer_size; i++)
                    local_gradient[layer_no][i] = error_output[i] * layer_derivatives[layer_no][i];

                break;
            case 3: // tanh
                d_tanh(param->output_layer_size, layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < param->output_layer_size; i++)
                    local_gradient[layer_no][i] = error_output[i] * layer_derivatives[layer_no][i];

                break;
            case 4: // relu
                d_relu(param->output_layer_size, layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < param->output_layer_size; i++)
                    local_gradient[layer_no][i] = error_output[i] * layer_derivatives[layer_no][i];

                break;
            case 5: // softmax
                d_softmax(param->output_layer_size, layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < param->output_layer_size; i++)
                    local_gradient[layer_no][i] = error_output[i] * layer_derivatives[layer_no][i];

                break;
            default:
                printf("Calculate local gradient: Invalid output activation function\n");
                exit(0);
                break;
        }

        // Free the memory allocated in Heap
        free(error_output);
    }
    else { // If hidden layer
        // Calculate the layer derivative for all units in the layer
        // Calculate local gradient
        int j;
        switch (param->hidden_activation_functions[layer_no-1]) {
            case 1: // identity
                d_identity(layer_sizes[layer_no], layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < layer_sizes[layer_no]; i++) {
                    double error = 0.0;
                    for (j = 0; j < layer_sizes[layer_no+1]; j++)
                        error += local_gradient[layer_no+1][j] * param->weight[layer_no][i][j];

                    local_gradient[layer_no][i] = error * layer_derivatives[layer_no][i];
                }

                break;
            case 2: // sigmoid
                d_sigmoid(layer_sizes[layer_no], layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < layer_sizes[layer_no]; i++) {
                    double error = 0.0;
                    for (j = 0; j < layer_sizes[layer_no+1]; j++)
                        error += local_gradient[layer_no+1][j] * param->weight[layer_no][i][j];

                    local_gradient[layer_no][i] = error * layer_derivatives[layer_no][i];
                }

                break;
            case 3: // tanh
                d_tanh(layer_sizes[layer_no], layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < layer_sizes[layer_no]; i++) {
                    double error = 0.0;
                    for (j = 0; j < layer_sizes[layer_no+1]; j++)
                        error += local_gradient[layer_no+1][j] * param->weight[layer_no][i][j];

                    local_gradient[layer_no][i] = error * layer_derivatives[layer_no][i];
                }

                break;
            case 4: // relu
                d_relu(layer_sizes[layer_no], layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < layer_sizes[layer_no]; i++) {
                    double error = 0.0;
                    for (j = 0; j < layer_sizes[layer_no+1]; j++)
                        error += local_gradient[layer_no+1][j] * param->weight[layer_no][i][j];

                    local_gradient[layer_no][i] = error * layer_derivatives[layer_no][i];
                }

                break;
            case 5: // softmax
                d_softmax(layer_sizes[layer_no], layer_inputs[layer_no], layer_outputs[layer_no], layer_derivatives[layer_no]);

                for (i = 0; i < layer_sizes[layer_no]; i++) {
                    double error = 0.0;
                    for (j = 0; j < layer_sizes[layer_no+1]; j++)
                        error += local_gradient[layer_no+1][j] * param->weight[layer_no][i][j];

                    local_gradient[layer_no][i] = error * layer_derivatives[layer_no][i];
                }

                break;
            default:
                printf("Invalid hidden activation function\n");
                exit(0);
                break;
        }
    }

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers; i++)
        free(layer_derivatives[i]);

    free(layer_derivatives);

}

void back_propagation(parameters* param, int training_example, int n_layers, int* layer_sizes, double** layer_inputs, double** layer_outputs) {
    /* ------------------ Expected output ----------------------------------------*/
    // Get the expected output from the data matrix
    // Create memory for the expected output array
    // Initialized to zero's
    double* expected_output = (double*)calloc(param->output_layer_size, sizeof(double));

    // Make the respective element in expected_output to 1 and rest all 0
    // Ex: If y = 3 and output_layer_size = 4 then expected_output = [0, 0, 1, 0]
    if (param->output_layer_size == 1)
        expected_output[0] = param->data_train[training_example][param->feature_size-1];
    else 
        expected_output[(int)(param->data_train[training_example][param->feature_size-1] - 1)] = 1;

    /* ---------------------- Weight correction Memory allocation ----------------------------------- */
    // Create memory for the weight_correction matrices between layers
    // weight_correction is a pointer to the array of 2D arrays between the layers
    double*** weight_correction = (double***)calloc(n_layers-1, sizeof(double**));

    // Each 2D array between two layers i and i+1 is of size ((layer_size[i]+1) x layer_size[i+1])
    // The weight_correction matrix includes weight corrections for the bias terms too
    int i;
    for (i = 0; i < n_layers-1; i++)
        weight_correction[i] = (double**)calloc(layer_sizes[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            weight_correction[i][j] = (double*)calloc(layer_sizes[i+1], sizeof(double));

    /* --------------------- Local Gradient memory allocation --------------------------------------*/
    // Create memory for local gradient (delta) for each layer
    double** local_gradient = (double**)calloc(n_layers, sizeof(double*));

    for (i = 0; i < n_layers; i++)
        local_gradient[i] = (double*)calloc(layer_sizes[i], sizeof(double));

    /*----------- Calculate weight corrections for all layers' weights -------------------*/
    // Weight correction for the output layer
    calculate_local_gradient(param, n_layers-1, n_layers, layer_sizes, layer_inputs, layer_outputs, expected_output, local_gradient);
    for (i = 0; i < param->output_layer_size; i++)
        for (j = 0; j < layer_sizes[n_layers-2]+1; j++)
            weight_correction[n_layers-2][j][i] = (param->learning_rate) * local_gradient[n_layers-1][i] * layer_outputs[n_layers-2][j];

    // Weight correction for the hidden layers
    int k;
    for (i = n_layers-2; i >= 1; i--) {
        calculate_local_gradient(param, i, n_layers, layer_sizes, layer_inputs, layer_outputs, expected_output, local_gradient);

        for (j = 0; j < layer_sizes[i]; j++) 
            for (k = 0; k < layer_sizes[i-1]+1; k++)
                weight_correction[i-1][k][j] = (param->learning_rate) * local_gradient[i][j] * layer_outputs[i-1][k];
    }

    /*----------------- Update the weights -------------------------------------*/
    for (i = 0; i < n_layers-1; i++) {
        for (j = 0; j < layer_sizes[i]+1; j++) {
            for (k = 0; k < layer_sizes[i+1]; k++) {
                param->weight[i][j][k] -= weight_correction[i][j][k];
            }
        }
    }


    // Free the memory allocated in Heap
    for (i = 0; i < n_layers; i++)
        free(local_gradient[i]);

    free(local_gradient);

    for (i = 0; i < n_layers - 1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            free(weight_correction[i][j]);

    for (i = 0; i < n_layers - 1; i++)
        free(weight_correction[i]);

    free(weight_correction);

    free(expected_output);
}
