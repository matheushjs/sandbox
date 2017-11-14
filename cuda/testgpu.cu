#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BLOCKS 1024
#define THREADS 1024
#define SIZE BLOCKS*THREADS*16

__global__
void testgpu(int *memInt, int times){
	int i;
	for(i = 0; i < times; i++)
		*memInt += (*memInt)*i;
}

int main(int argc, char *argv[]){
	int *gpuInt;

	dim3 block(1024, 1);
	dim3 grid(1024, 1024);

	cudaMalloc( (void **) &gpuInt, sizeof(int));
	
	// printf("Test 1\n");
	// testgpu<<<grid, block>>>(gpuInt, 800000);
	
	printf("Test 2\n");
	testgpu<<<1, 1>>>(gpuInt, 1024 * 1024 * 1024);

	cudaFree(gpuInt);

	return 0;
}
