#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct
{
    float *data;    // pointer to data
    int ndim;       // number of dimensions
    int64_t *shape; // array of dimension sizes
    size_t size;    // total number of elements
} Tensor;

Tensor *tensor_new(int ndim, const int64_t *shape);
void tensor_free(Tensor *t);
void tensor_print(const Tensor *t);

#endif;