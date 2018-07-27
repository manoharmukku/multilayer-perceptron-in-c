#ifndef READ_CSV_H
#define READ_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1048576 // 2^20

void read_csv(char*, int, int, double**);

#endif