#include "ops.h"
#include <stddef.h>
#include <stdint.h>

static int same_shape(const Tensor *a, const Tensor *b)
{
    if (!a || !b)
    {
        return 0;
    }
    if (a->ndim != b->ndim)
    {
        return 0;
    }
    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return 0;
        }
    }
    return 1;
}

int tensor_add(const Tensor *a, const Tensor *b, Tensor *out)
{
    if (!a || !b || !out || !a->data || !b->data || !out->data)
    {
        return -1;
    }
    if (!same_shape(a, b) || !same_shape(a, out))
    {
        return -1;
    }
    for (size_t i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    return 0;
}

int tensor_add_scalar(const Tensor *a, float v, Tensor *out)
{
    if (!a || !out || !a->data || !out->data)
    {
        return -1;
    }
    if (!same_shape(a, out))
    {
        return -1;
    }
    for (size_t i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + v;
    }
    return 0;
}

int tensor_sum(const Tensor *a, float *out)
{
    if (!a || !out || !a->data)
    {
        return -1;
    }
    double acc = 0.0;
    for (size_t i = 0; i < a->size; i++)
    {
        acc += a->data[i];
    }
    *out = (float)acc;
    return 0;
}

int tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C)
{
    if (!A || !B || !C || !A->data || !B->data || !C->data)
    {
        return -1;
    }
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2)
    {
        return -1;
    }

    int64_t m = A->shape[0];
    int64_t k = A->shape[1];

    if (B->shape[0] != k)
    {
        return -1;
    }

    int64_t n = B->shape[1];
    if (C->shape[0] != m || C->shape[1] != n)
    {
        return -1;
    }

    const float *a = A->data;
    const float *b = B->data;
    float *c = C->data;

    for (int64_t row = 0; row < m; row++)
    {
        size_t a_base = (size_t)row * (size_t)k;
        size_t c_base = (size_t)row * (size_t)n;
        for (int64_t col = 0; col < n; col++)
        {
            double acc = 0.0;
            for (int64_t p = 0; p < k; p++)
            {
                acc += (double)a[a_base + (size_t)p] *
                       (double)b[(size_t)p * (size_t)n + (size_t)col];
            }
            c[c_base + (size_t)col] = (float)acc;
        }
    }
    return 0;
}