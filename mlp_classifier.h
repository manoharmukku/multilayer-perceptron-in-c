#ifndef MLP_CLASSIFIER_H
#define MLP_CLASSIFIER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read_csv.h"
#include "parameters.h"

#define max(x, y) (x > y ? x : y)

void mlp_classifier(parameters*);

#endif