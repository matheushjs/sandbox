#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

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

	printf("A %lf\n", clock() / (double) CLOCKS_PER_SEC);
	cudaMalloc( (void **) &gpuInt, sizeof(int));
	printf("B %lf\n", clock() / (double) CLOCKS_PER_SEC);
	
	// printf("Test 1\n");
	// testgpu<<<grid, block>>>(gpuInt, 800000);
	
	printf("C %lf\n", clock() / (double) CLOCKS_PER_SEC);
	testgpu<<<8, 16>>>(gpuInt, 1024 * 1024 * 1024);
	printf("D %lf\n", clock() / (double) CLOCKS_PER_SEC);
	
	cudaFree(gpuInt);
	printf("E %lf\n", clock() / (double) CLOCKS_PER_SEC);

	return 0;
}
