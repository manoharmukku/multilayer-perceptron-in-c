/*
Author: Manohar Mukku
Date: 23.07.2018
Desc: Backpropagation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

typedef struct {
    int n_hidden;
    int* hidden_layers_size;
    int* hidden_activation_functions;
    float regularization_parameter;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
    int sample_size;
    int feature_size;
    double** data;
} parameters;

void back_propagation(parameters* param, int training_example, int n_layers, int** layer_sizes, double** layer_inputs, double** layer_outputs, double*** theta) {
    // Get the expected output from the data matrix
    // Create memory for the expected output array
    // Initialized to zero's
    double* expected_output = (double*)calloc(output_layer_size, sizeof(double));

    // Make the respective element in expected_output to 1 and rest all 0
    // Ex: If y = 3 and output_layer_size = 4 then expected_output = [0, 0, 1, 0]
    if (output_layer_size == 1)
        expected_output[0] = param->data[training_example][param->feature_size-1];
    else 
        expected_output[param->data[training_example][param->feature_size-1] - 1] = 1;

        

    free(expected_output);
}
