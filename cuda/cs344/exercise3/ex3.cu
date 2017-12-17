#include <stdio.h>
#include "timer.h"
#include "utils.h"

const int BLOCKSIZE = 128;
const int NUMBLOCKS = 1;
const int N         = BLOCKSIZE*NUMBLOCKS;

__global__ void foo(float out[], float A[], float B[], float C[], float D[], float E[]){
	int i = threadIdx.x + blockIdx.x*blockDim.x; 

	out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
}

__global__ void bar(float out[], float in[]) {
	int i = threadIdx.x + blockIdx.x*blockDim.x; 
	int tid = threadIdx.x;

	__shared__ float shmem[BLOCKSIZE + 4];
	shmem[tid + 2] = in[i];
	
	// If there are problems with trash values:
	//shmem[0] = 0;
	//shmem[1] = 0;
	//shmem[BLOCKSIZE+2] = 0;
	//shmem[BLOCKSIZE+3] = 0;
	
	__syncthreads();

	// This will work only because of how threads are organized in warps
	// Thread 1 will also wait for shmem[1] to be filled
	if(tid == 0 && i != 0 && i != 1){
		shmem[0] = in[i-2];
		shmem[1] = in[i-1];
	} else if(tid == BLOCKSIZE-1 && i != N-1 && i != N-2){
		// All threads in same warp as thread BLOCKSIZE-1 will pass here.
		shmem[BLOCKSIZE+2] = in[i+1];
		shmem[BLOCKSIZE+3] = in[i+2];
	}


	out[i] = (shmem[tid] + shmem[tid+1] + shmem[tid+2] + shmem[tid+3] + shmem[tid+4]) / 5.0f;
}

void cpuFoo(float out[], float A[], float B[], float C[], float D[], float E[]) {
	for (int i=0; i<N; i++)
	{
		out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
	}
}

void cpuBar(float out[], float in[]) {
	// ignore the boundaries
	for (int i=2; i<N-2; i++)
	{
		out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
	}
}

int compareArrays(float *a, float *b, int N){
	for(int i = 2; i < N-2; i++)
		if(a[i] != b[i]){
			printf("Failed. %f != %f\n", a[i], b[i]);
			return 1;
		}
	printf("Correct.\n");
	return 0;
}

int main(int argc, char **argv) {
	// declare and fill input arrays for foo() and bar()
	float fooA[N], fooB[N], fooC[N], fooD[N], fooE[N], barIn[N];
	for (int i=0; i<N; i++) 
	{
		fooA[i] = i; 
		fooB[i] = i+1;
		fooC[i] = i+2;
		fooD[i] = i+3;
		fooE[i] = i+4;
		barIn[i] = 2*i; 
	}
	// device arrays
	int numBytes = N * sizeof(float);
	float *d_fooA;	 	cudaMalloc(&d_fooA, numBytes);
	float *d_fooB; 		cudaMalloc(&d_fooB, numBytes);
	float *d_fooC;	 	cudaMalloc(&d_fooC, numBytes);
	float *d_fooD; 		cudaMalloc(&d_fooD, numBytes);
	float *d_fooE; 		cudaMalloc(&d_fooE, numBytes);
	float *d_barIn; 	cudaMalloc(&d_barIn, numBytes);
	cudaMemcpy(d_fooA, fooA, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooB, fooB, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooC, fooC, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooD, fooD, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fooE, fooE, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_barIn, barIn, numBytes, cudaMemcpyHostToDevice);	

	// output arrays for host and device
	float fooOut[N], barOut[N], *d_fooOut, *d_barOut;
	cudaMalloc(&d_fooOut, numBytes);
	cudaMalloc(&d_barOut, numBytes);

	// declare and compute reference solutions
	float ref_fooOut[N], ref_barOut[N]; 
	cpuFoo(ref_fooOut, fooA, fooB, fooC, fooD, fooE);
	cpuBar(ref_barOut, barIn);

	// launch and time foo and bar
	GpuTimer fooTimer, barTimer;
	fooTimer.Start();
	foo<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_fooOut, d_fooA, d_fooB, d_fooC, d_fooD, d_fooE);
	fooTimer.Stop();
	
	barTimer.Start();
	bar<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_barOut, d_barIn);
	barTimer.Stop();

	cudaMemcpy(fooOut, d_fooOut, numBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost);
	printf("foo<<<>>>(): %g ms elapsed. Verifying solution...\n", fooTimer.Elapsed());
	compareArrays(ref_fooOut, fooOut, N);
	printf("bar<<<>>>(): %g ms elapsed. Verifying solution...\n", barTimer.Elapsed());
	compareArrays(ref_barOut, barOut, N);
}
