#include <stdio.h>
#include "utils.h"

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
	// We will use 1D block organization and 1D thread organization
	// For each block, we calculate the range in the vector it has to work with
	// Then we reduce it
	// Then we reduce all blocks
	// Any vector element out-of-range gets the --identity-- value
	// TODO: We will waste half of the threads allocated for now. Fix this later.

	// DOC
	// - number of threads must be multiple of 2
	// - total number of threads must exceed the vector size
	// - 'max' must be initialized to a low value (identity for max)
	// - 'min' must be initialized to a high value (identity for min)
	// - 'lock' must be initialized to 0

	int beg = blockIdx.x * blockDim.x;
	int size = blockDim.x;

	// Get shared memory (intermediate values)
	// 'size' is expected to be at most 1024, so we're allocating at most 4kB of shared memory
	__shared__ float maxVec[REDUCE_BLOCKDIM/2];
	__shared__ float minVec[REDUCE_BLOCKDIM/2];

	// Copy elements from vec to shared memory, while doing the first iteration of the reduce
	int step = size >> 1;
	if(threadIdx.x < step){
		int right, left;
		right = beg + threadIdx.x + step;
		left = beg + threadIdx.x;

		// Elements out of bounds receive an identity value.
		if(left >= vecSize){
			maxVec[threadIdx.x] = FLOAT_MIN;
			minVec[threadIdx.x] = FLOAT_MAX;
		} else if(right >= vecSize){
			maxVec[threadIdx.x] = vec[left];
			minVec[threadIdx.x] = vec[left];
		} else {
			maxVec[threadIdx.x] = maxKer(vec[beg + threadIdx.x], vec[beg + threadIdx.x + step]);
			minVec[threadIdx.x] = minKer(vec[beg + threadIdx.x], vec[beg + threadIdx.x + step]);
		}
	}

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
	// - Naive histogram
	// - Will rely on atomic operations upon the global memory
	// - 1D block and 1D thread organizations. Any numbers will do
	int beg = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; // Total number of threads

	while(beg < vecSize){
		int bin = (vec[beg] - min) / range * numBins;
		atomicAdd(&bins[bin], 1);
		beg += stride;
	}
}

/*****************************************/
/**********SCAN***************************/
/*****************************************/

__global__
void scan(const int * const bins, int * const accSum, int numBins){
	// DOC
	// - Hillis & Steele algorithm
	// - inclusive scan
	// - Single block, 1D threads
	// - Total number of threads should equal at least numBins/2
	// - numBins is assumed to be a power of 2
	int idx = threadIdx.x;

	// Copy bins to accSum
	int i = idx;
	while(i < numBins){
		accSum[i] = bins[i];
		i += blockDim.x;
	}

	__syncthreads();

	int step = 1;
	while(step != numBins){
		int next = idx + step;
		int val = accSum[idx] + accSum[next];
		__syncthreads();

		if(next < numBins)
			accSum[next] = val;

		__syncthreads();
		step <<= 1;
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	int vecSize = numRows * numCols;
	int numBlocks = vecSize / (double) REDUCE_BLOCKDIM + 1;

	printf("Rows: %lu\nCols: %lu\nPixels: %lu\nBins: %lu\n\n", numRows, numCols, numCols*numRows, numBins);

	/*
	float *vector = (float *) malloc(sizeof(float) * vecSize);
	cudaMemcpy(vector, d_logLuminance, sizeof(float) * vecSize, cudaMemcpyDeviceToHost);
	float max = 0, min = 275;
	for(int i = 0; i < vecSize; i++){
		max = std::max(max, vector[i]);
		min = std::min(min, vector[i]);
	}
	printf("%f %f\n", min, max);
	free(vector);
	*/
	// Max should be 2.189105
	// Min should be -4

	// Get device memory
	float *d_max, *d_min;
	int *d_lock;

	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_lock, sizeof(int)));
	checkCudaErrors(cudaMemset(d_max, FLOAT_MIN, sizeof(float)));
	checkCudaErrors(cudaMemset(d_min, FLOAT_MAX, sizeof(float)));
	checkCudaErrors(cudaMemset(d_lock, 0, sizeof(int)));

	reduce_maxmin<<<numBlocks, REDUCE_BLOCKDIM>>>(d_logLuminance, vecSize, d_max, d_min, d_lock);
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_lock));

	printf("Min: %f, Max: %f\n", min_logLum, max_logLum);




	int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));
	histogram
		<<<numBlocks, REDUCE_BLOCKDIM>>>
		(d_logLuminance, d_bins, min_logLum, max_logLum - min_logLum, vecSize, numBins);
	

	// We will reuse the d_bins vector
	scan<<<1, 512>>>(d_bins, (int *) d_cdf, numBins);
	checkCudaErrors(cudaFree(d_bins));
}
