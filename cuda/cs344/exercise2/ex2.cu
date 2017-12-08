#include <stdio.h>
#include "timer.h"
#include "utils.h"

const int N= 1024;	// matrix size will be NxN
const int K= 32;		// TODO, set K to the correct value and tile size will be KxK


// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
	// TODO
	__shared__ float mid[K][K];
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	mid[threadIdx.x][threadIdx.y] = in[col + row*N];
	__syncthreads();

	col = blockDim.y * blockIdx.y + threadIdx.x;
	row = blockDim.x * blockIdx.x + threadIdx.y;
	out[col + row*N] = mid[threadIdx.y][threadIdx.x];
}

void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

void fill_matrix(float *in, int N){
	for(int i = 0; i < N*N; i++)
		in[i] = i;
}

int compare_matrices(float *a, float *b, int N){
	for(int i = 0; i < N*N; i++)
		if(a[i] != b[i]) return 1;
	return 0;
}

void print_matrix(float *a, int N){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%d ", (int) a[i*N + j]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);

	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);

	fill_matrix(in, N);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

/*  
 * Now time each kernel and verify that it produces the correct result.
 *
 * To be really careful about benchmarking purposes, we should run every kernel once
 * to "warm" the system and avoid any compilation or code-caching effects, then run 
 * every kernel 10 or 100 times and average the timings to smooth out any variance. 
 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
 */

	dim3 blocks(N/K,N/K);	//TODO, you need to set the proper blocks per grid
	dim3 threads(K,K);	//TODO, you need to set the proper threads per block

	timer.Start();
	transpose_parallel_per_element_tiled<<<blocks,threads>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}
