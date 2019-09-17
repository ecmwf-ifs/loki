/**
 * Document: MaxCompiler Tutorial (maxcompiler-tutorial.pdf)
 * Chapter: 4      Example: 1      Name: Pass-Through
 * MaxFile name: PassThrough
 * Summary:
 * 	Take a stream of values, send it through the dataflow engine
 *  and return back the same stream.
 */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <MaxSLiCInterface.h>
#include "PassThrough.h"

int check(uint32_t *dataOut, uint32_t *expected, int size)
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


uint32_t dataIn[1024];
uint32_t dataOut[1024];
uint32_t expected[1024];
const int size = 1024;

int main()
{
	for (int i = 0; i < size; i++) {
		dataIn[i] = i + 1;
		dataOut[i] = 0;
	}

	PassThroughCPU(size, dataIn, expected);

	printf("Running DFE.\n");
	PassThrough(size, dataIn, size * sizeof(uint32_t), dataOut, size * sizeof(uint32_t));

	int status = check(dataOut, expected, size);
	if (status)
		printf("Test failed.\n");
	else
		printf("Test passed OK!\n");
	return status;
}
