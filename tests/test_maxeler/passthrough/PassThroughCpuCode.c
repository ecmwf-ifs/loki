/**
 * Document: MaxCompiler Tutorial (maxcompiler-tutorial.pdf)
 * Chapter: 4      Example: 1      Name: Pass-Through
 * MaxFile name: PassThrough
 * Summary:
 *  Take a stream of values, send it through the dataflow engine
 *  and return back the same stream.
 */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <MaxSLiCInterface.h>
#include "passthrough.h"

int check(int size, uint32_t *dataOut, uint32_t *expected)
{
    int status = 0;
    for (int i = 0; i < size; i++) {
        if (dataOut[i] != expected[i]) {
            fprintf(stderr, "Output data @ %d = %d (expected %d)\n",
                i, dataOut[i], expected[i]);
            status = 1;
        }
    }

    return status;
}

void PassThroughCPU(int size, uint32_t *dataIn, uint32_t *dataOut)
{
    for (int i = 0 ; i < size ; i++) {
        dataOut[i] = dataIn[i];
    }
}

void initializeData(int size, uint32_t *dataIn, uint32_t *dataOut)
{
    for (int i = 0; i < size; i++) {
        dataIn[i] = i + 1;
        dataOut[i] = 0;
    }
}
    

int main()
{
    int size = 1024;
    uint32_t dataIn = (uint32_t*) malloc(size * sizeof(uint32_t));
    uint32_t dataOut = (uint32_t*) malloc(size * sizeof(uint32_t));
    uint32_t expected = (uint32_t*) malloc(size * sizeof(uint32_t));

    initializeData(size, dataIn, dataOut);
    PassThroughCPU(size, dataIn, expected);

    printf("Running DFE.\n");
    passthrough(size, dataIn, size * sizeof(uint32_t), dataOut, size * sizeof(uint32_t));

    int status = check(size, dataOut, expected);
    if (status)
        printf("Test failed.\n");
    else
        printf("Test passed OK!\n");

    free(expected);
    free(dataOut);
    free(dataIn);

    return status;
}
