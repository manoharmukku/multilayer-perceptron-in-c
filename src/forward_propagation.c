/*
Author: Manohar Mukku
Date: 20.07.2018
Desc: Feedforward propagation
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

void forward_propagation() {
    
}
