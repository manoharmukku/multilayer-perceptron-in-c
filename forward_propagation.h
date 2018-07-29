#ifndef FORWARD_PROPAGATION_H
#define FORWARD_PROPAGATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "parameters.h"

#define max(x, y) (x > y ? x : y)

void forward_propagation(parameters*, int, int, int*, double**, double**, double***);

#endif