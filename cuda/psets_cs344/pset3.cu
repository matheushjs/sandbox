#include <stdio.h>
#include "utils.h"
#include "timer.h"

#define REDUCE_BLOCKDIM 1024 // Must be power of 2
#define FLOAT_MIN -275.0f
#define FLOAT_MAX 275.0f

/*****************************************/
/**********REDUCE MAX MIN*****************/
/*****************************************/

__device__
float maxKer(float a, float b){
	if(a > b) return a;
	else return b;
}

__device__
float minKer(float a, float b){
	if(a < b) return a;
	else return b;
}

__global__
void reduce_maxmin(const float * const vec, int vecSize, float * const max, float * const min, int * const lock){
	// DOC
	// - 1D block and thread organization.
	// - number of threads must be power of 2
	// - total number of threads must be at least (vector_size)/2 rounded up.
	// - 'max' must be initialized to a low value (identity for max)
	// - 'min' must be initialized to a high value (identity for min)
	// - 'lock' must be initialized to 0

	int beg = blockIdx.x * blockDim.x * 2; // Each block of X threads will reduce a block of 2*X data elements.
	int size = blockDim.x * 2;

	// Get shared memory (intermediate values)
	// We're allocating at most 8kB of shared memory
	__shared__ float maxVec[REDUCE_BLOCKDIM];
	__shared__ float minVec[REDUCE_BLOCKDIM];

	// Copy elements from vec to shared memory, while doing the first iteration of the reduce
	// This just a single step, so we won't bother with having many if-statements.
	int step = size >> 1; // First step is half of the full block size
	                      // So if there are 1024 threads, block is 2048 data elements, first step is 1024.
	if(threadIdx.x < step){
		const int right = beg + threadIdx.x + step; // positions in the original vector
		const int left = beg + threadIdx.x;

		// Elements out of bounds (in original vec) receive an identity value (in shared vec).
		if(left >= vecSize){ // If left is out of bounds, right is too
			maxVec[threadIdx.x] = FLOAT_MIN;
			minVec[threadIdx.x] = FLOAT_MAX;
		} else if(right >= vecSize){ // left isn't out of bounds, so we copy it
			maxVec[threadIdx.x] = vec[left];
			minVec[threadIdx.x] = vec[left];
		} else {
			maxVec[threadIdx.x] = maxKer(vec[left], vec[right]);
			minVec[threadIdx.x] = minKer(vec[left], vec[right]);
		}
	}

	// Sync after copy
	__syncthreads();

	// Reduce stuff in shared memory
	for(step >>= 1; step > 0; step >>= 1){
		if(threadIdx.x < step){
			if(maxVec[threadIdx.x + step] > maxVec[threadIdx.x])
				maxVec[threadIdx.x] = maxVec[threadIdx.x + step];

			if(minVec[threadIdx.x + step] < minVec[threadIdx.x])
				minVec[threadIdx.x] = minVec[threadIdx.x + step];
		}
	}

	// Atomically write results
	if(threadIdx.x == 0){
		while( atomicCAS(lock, 0, 1) != 0 );
		*max = maxKer(*max, maxVec[0]);
		*min = minKer(*min, minVec[0]);
		*lock = 0;
	}
}

/*****************************************/
/**********HISTOGRAM**********************/
/*****************************************/

__global__
void histogram(const float * const vec, int * const bins, const float min, const float range, const int vecSize, const int numBins){
	// DOC
	// - 1D block and 1D thread organizations.
	// - Total number of threads can be anything
	// - More blocks mean more shared memory usage
	// - Each block will allocate shared memory that can hold the whole bin vector.
	// - Each block will be responsible for a section of the vector.
	// - After finding the histogram for its own vector, will sum into 'bins'.
	// - Dynamic shared memory should be allocated, with size sizeof(int) * numBins

	extern __shared__ int loc_bins[];

	if(threadIdx.x < numBins)
		loc_bins[threadIdx.x] = 0;

	__syncthreads();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while(idx < vecSize){
		int bin = (numBins * (vec[idx] - min) / (range + 1E-5));
		atomicAdd(&loc_bins[bin], 1);
		idx += stride;
	}

	__syncthreads();

	if(threadIdx.x < numBins)
		atomicAdd(&bins[threadIdx.x], loc_bins[threadIdx.x]);
}

/*****************************************/
/**********SCAN***************************/
/*****************************************/

__global__
void scan(const int * const bins, int * const accSum, int numBins){
	// DOC
	// - Hillis & Steele algorithm
	// - inclusive scan, but converted to exclusive before returning
	// - Single block, 1D threads
	// - Total number of threads should equal at least numBins/2
	// - numBins is assumed to be a power of 2
	// - Requires dynamic shared memory of size sizeof(int) * numBins

	extern __shared__ int loc_bins[];
	const int idx = threadIdx.x;
	const int stride = blockDim.x; // Assuming single block

	// Copy bins to loc_bins
	int i = idx;
	while(i < numBins){
		loc_bins[i] = bins[i];
		i += stride;
	}

	__syncthreads();

	// Scan phase
	int step = 1;
	while(step != numBins){
		const int next = idx + step;
		const int val = loc_bins[idx];

		if(next < numBins)
			loc_bins[next] = val + loc_bins[next];

		__syncthreads();
		step <<= 1;
	}

	// Copy to accSum, converting to exclusive scan
	step = idx;
	while(step < (numBins-1)){
		accSum[step + 1] = loc_bins[step];
		step += stride;
	}

	if(idx == 0)
		accSum[0] = 0;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	GpuTimer timer;

	int vecSize = numRows * numCols;

	printf("Rows: %lu\nCols: %lu\nPixels: %lu\nBins: %lu\n\n", numRows, numCols, numCols*numRows, numBins);

	// Get device memory
	float *d_max, *d_min;
	int *d_lock;

	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_lock, sizeof(int)));
	checkCudaErrors(cudaMemset(d_max, FLOAT_MIN, sizeof(float)));
	checkCudaErrors(cudaMemset(d_min, FLOAT_MAX, sizeof(float)));
	checkCudaErrors(cudaMemset(d_lock, 0, sizeof(int)));

	int numThreads = REDUCE_BLOCKDIM;
	int numBlocks = vecSize / (2*numThreads) + 1;
	timer.Start();
	reduce_maxmin<<<numBlocks, numThreads>>>(d_logLuminance, vecSize, d_max, d_min, d_lock);
	timer.Stop();
	printf("Elapsed for reduce: %f msec\n", timer.Elapsed());

	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_lock));
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	
	int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));

	numThreads = 256;
	numBlocks = 32; // For some reason, 1/2 blocks seem optimal for this case.
	               // Probably related to the gigantic amount of pixels falling into the very same bin, each time
	timer.Start();
	histogram
		<<<numBlocks, numThreads, sizeof(int) * numBins>>>
		(d_logLuminance, d_bins, min_logLum, max_logLum - min_logLum, vecSize, numBins);
	timer.Stop();
	printf("Elapsed for histogram: %f msec\n", timer.Elapsed());
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());




	numThreads = 1024;
	numBlocks = 1;

	// We will reuse the d_bins vector
	timer.Start();
	scan
		<<<numBlocks, numThreads, sizeof(int) * numBins>>>
		(d_bins, (int *) d_cdf, numBins);
	timer.Stop();
	printf("Elapsed for scan: %f msec\n", timer.Elapsed());
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_bins));
	printf("\n\n");
}
