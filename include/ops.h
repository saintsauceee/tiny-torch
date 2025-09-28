#ifndef OPS_H
#define OPS_H
#include "tensor.h"

int tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
int tensor_add_scalar(const Tensor *a, float v, Tensor *out);
int tensor_sum(const Tensor *a, float *out);
int tensor_matmul(const Tensor *A, const Tensor *B, const Tensor *C);

#endif