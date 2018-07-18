/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: MLP Classifier
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n_hidden;
    int* hidden_layer_size;
    int hidden_activation_function;
    float regularization_parameter;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
    int sample_size;
    int feature_size;
    double** data;
} parameters;

void MLPClassifier(parameters* params) {
    
}
