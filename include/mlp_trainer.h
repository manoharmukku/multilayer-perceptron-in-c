#ifndef MLP_TRAINER_H
#define MLP_TRAINER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "forward_propagation.h"
#include "back_propagation.h"
#include "parameters.h"

void mlp_trainer(parameters* param, int*);

#endif