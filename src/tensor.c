#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

// allocate a new tensor
Tensor *tensor_new(int ndim, const int64_t *shape)
{
    Tensor *t = malloc(sizeof(Tensor));
    if (!t)
    {
        return NULL;
    }

    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(int64_t));
    if (!t->shape)
    {
        free(t);
        return NULL;
    }

    size_t total = 1;
    for (int i = 0; i < ndim; i++)
    {
        t->shape[i] = shape[i];
        total *= shape[i];
    }
    t->size = total;

    t->data = calloc(total, sizeof(float)); // init to zeroes
    if (!t->data)
    {
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

// free tensor memory
void tensor_free(Tensor *t)
{
    if (!t)
    {
        return;
    }
    free(t->data);
    free(t->shape);
    free(t);
}

// print out tensor
void tensor_print(const Tensor *t)
{
    printf("Tensor Shape: (");
    for (int i = 0; i < t->ndim; i++)
    {
        printf("%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1)
        {
            printf(", ");
        }
    }
    printf(")\nData: [");
    for (size_t i = 0; i < t->size; i++)
    {
        printf("%.2f", t->data[i]);
        if (i < t->size - 1)
        {
            print(", ");
        }
    }
    printf("]\n");
}