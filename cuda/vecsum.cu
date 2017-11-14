#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCKS 1024
#define THREADS 1024
#define SIZE BLOCKS*THREADS*16

void print(int *vec){
	for(int i = 0; i < SIZE; i++)
		printf("%d ", vec[i]);
	printf("\n");
}

int *get_vector(int n){
	int *res = (int *) malloc(sizeof(int) * n);
	for(int i = 0; i < n; i++)
		res[i] = 1;
	return res;
}

__global__
void add(int *vec1, int *vec2, int times){
	const int beg = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j;
	for(i = beg; i < SIZE; i += blockDim.x * gridDim.x)
		for(j = 0; j < times; j++)
			vec2[i] = vec2[i] + vec1[i];
	//__syncthreads();
}

int main(int argc, char *argv[]){
	if(SIZE > 100 * 1024 * 1024); // If over 100 megabytes
	printf("Size: %lf Mb\n", SIZE / (double) 1024 / (double) 1024);

	int *vec1 = get_vector(SIZE);
	int *vec2 = get_vector(SIZE);
	void *gpuVec1, *gpuVec2;

	cudaMalloc( &gpuVec1, sizeof(int) * SIZE );
	cudaMalloc( &gpuVec2, sizeof(int) * SIZE );
	cudaMemcpy( gpuVec1, vec1, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy( gpuVec2, vec2, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

	printf("First element: %d\n", vec2[0]);

	add<<<BLOCKS, THREADS>>>((int*) gpuVec1, (int*) gpuVec2, 10000);

	cudaMemcpy( vec2, gpuVec2, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
	//print(vec2);
	printf("First element: %d\n", vec2[0]);

	cudaFree(gpuVec1);
	cudaFree(gpuVec2);
	free(vec1);
	free(vec2);
	return 0;
}
