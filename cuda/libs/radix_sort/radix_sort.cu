/////////
#include <iostream>
#include <algorithm>
#include "timer.h"
/////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "blelloch_scan.h"

#define NBITS 4
#define MASK 0xF

// Returns the idx-th group of bits from 'num'
__device__ static inline
unsigned int get_group(unsigned int num, int whichGroup){
	return (num >> (whichGroup * NBITS)) & MASK;
}

// Number of threads allocated must be higher than vecSize
__global__ static
void make_predicate(const unsigned int * const d_vec, unsigned int * const d_out, int vecSize, int bitGroup, unsigned int groupVal){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if(myIdx < vecSize)
		d_out[myIdx] = get_group(d_vec[myIdx], bitGroup) == groupVal;
}

// Number of threads allocated must be higher than vecSize
__global__ static
void scatter_elements(const unsigned int * const d_vec,
                      const unsigned int * const d_xscan,
                      unsigned int * const d_out,
                      unsigned int * d_baseIndex,
                      int vecSize)
{
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if(myIdx >= vecSize - 1) return; // Last element is a special case
	if(d_xscan[myIdx] != d_xscan[myIdx + 1])
		d_out[d_baseIndex[0] + d_xscan[myIdx]] = d_vec[myIdx];
}

// 1 thread only
__global__ static
void scatter_last( // Will scatter last element if needed, and update d_baseIndex as needed
		const unsigned int * const d_lastElem, // Last element of original vector
		unsigned int * const d_out,             // output vector
		unsigned int * const d_lastScan,       // Last element of the xscan vector
		unsigned int * const d_baseIndex,      // baseIndex
		int bitGroup,                          // bitGroup being analyzed (last 4 bits, second last etc...)
		unsigned int groupVal)
{                // value being analyzed in the the bit group (0, 1, 2...)
	const int lastElem = *d_lastElem;
	const int mustScatter = get_group(lastElem, bitGroup) == groupVal;
	const int outIndex = *d_baseIndex + *d_lastScan;

	if(mustScatter)
		d_out[outIndex] = lastElem;

	*d_baseIndex = outIndex + mustScatter; // If last element is of given group, we must add 1
}

void print_device(unsigned int *d_vec, int vecSize){
	int *vec = (int *) malloc(vecSize * sizeof(int));
	cudaMemcpy(vec, d_vec, sizeof(int) * vecSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < vecSize; i++)
		std::cout << vec[i] << " ";
	std::cout << "\n";
	free(vec);
}

// Assumes vector's size is a power of 2
void d_radix_sort(unsigned int **d_vec_p, int vecSize){
	const int nGroups = (sizeof(int) * 8) / NBITS;
	const int maxGroupVal = 1 << NBITS;

	unsigned int *d_vec = *d_vec_p;

	unsigned int *d_out, *d_baseIndex;
	cudaMalloc(&d_out, sizeof(int) * vecSize);
	cudaMalloc(&d_baseIndex, sizeof(int));

	cudaStream_t streams[4];
	unsigned int *d_pred[4];
	for(int i = 0; i < 4; i++){
		cudaStreamCreate(streams + i);
		cudaMalloc(d_pred + i, sizeof(int) * vecSize);
	}

	for(int group = 0; group < nGroups; group++){
		const int nThreads = 256;
		const int nBlocks = std::ceil(vecSize / (double) nThreads);

		// Reset base index
		cudaMemset(d_baseIndex, 0, sizeof(int));
		cudaDeviceSynchronize();

		for(int val = 0; val < maxGroupVal; val += 4){

			// Get predicates
			// Scan predicates
			for(int i = 0; i < 4; i++){
				if(val + i < maxGroupVal){
					make_predicate<<<nBlocks, nThreads, 0, streams[i]>>>(d_vec, d_pred[i], vecSize, group, val+i);
				}
			}
			for(int i = 0; i < 4; i++){
				if(val + i < maxGroupVal){
					xscan(d_pred[i], vecSize, streams[i]);
				}
			}

			// Sequentially Scatter and scatter last element
			for(int i = 0; i < 4; i++){
				if(val + i < maxGroupVal){
					cudaStreamSynchronize(streams[i]);
					scatter_elements<<<nBlocks, nThreads>>>(d_vec, d_pred[i], d_out, d_baseIndex, vecSize);
					scatter_last<<<1,1>>>(d_vec + vecSize - 1, d_out, d_pred[i] + vecSize - 1, d_baseIndex, group, val+i);
				}
			}
		}

		std::swap(d_vec, d_out);
	}

	for(int i = 0; i < 4; i++){
		cudaStreamDestroy(streams[i]);
		cudaFree(d_pred[i]);
	}
	cudaFree(d_out);
	cudaFree(d_baseIndex);

	*d_vec_p = d_vec;
}

// Assumes vector's size is a power of 2
void d_radix_sort_old(unsigned int **d_vec_p, int vecSize){
	const int nGroups = (sizeof(int) * 8) / NBITS;
	const int maxGroupVal = 1 << NBITS;

	unsigned int *d_vec = *d_vec_p;

	unsigned int *d_pred, *d_out, *d_baseIndex;
	cudaMalloc(&d_pred, sizeof(int) * vecSize);
	cudaMalloc(&d_out, sizeof(int) * vecSize);
	cudaMalloc(&d_baseIndex, sizeof(int));

	for(int group = 0; group < nGroups; group++){
		const int nThreads = 256;
		const int nBlocks = std::ceil(vecSize / (double) nThreads);

		// Reset base index
		cudaMemset(d_baseIndex, 0, sizeof(int));

		for(int val = 0; val < maxGroupVal; val++){

			// Get predicates
			// Scan predicates
			make_predicate<<<nBlocks, nThreads>>>(d_vec, d_pred, vecSize, group, val);
			xscan(d_pred, vecSize, (cudaStream_t) 0);

			// Scatter and scatter last element
			scatter_elements<<<nBlocks, nThreads>>>(d_vec, d_pred, d_out, d_baseIndex, vecSize);
			scatter_last<<<1,1>>>(d_vec + vecSize - 1, d_out, d_pred + vecSize - 1, d_baseIndex, group, val);
		}

		std::swap(d_vec, d_out);
	}

	cudaFree(d_pred);
	cudaFree(d_out);
	cudaFree(d_baseIndex);

	*d_vec_p = d_vec;
}


// 1 << 15 size
// Version 1: ~56.9 msecs
int main(int argc, char *argv[]){
	using std::cout;
	using std::sort;

	const int size = 1 << 15;
	unsigned int *h_vec = (unsigned int *) malloc(sizeof(int) * size);
	unsigned int *h_out = (unsigned int *) malloc(sizeof(int) * size);

	for(int i = 0; i < size; i++)
		h_vec[i] = i % 16;

	unsigned int *d_vec;
	cudaMalloc(&d_vec, sizeof(int) * size);
	cudaMemcpy(d_vec, h_vec, sizeof(int) * size, cudaMemcpyHostToDevice);

	GpuTimer timer;


	/* NEW APPROACH */
	timer.Start();
	d_radix_sort(&d_vec, size);
	timer.Stop();
	cout << "Elapsed (new): " << timer.Elapsed() << " msecs\n";


	/* OLD APPROACH */
	cudaMemcpy(d_vec, h_vec, sizeof(int) * size, cudaMemcpyHostToDevice);
	timer.Start();
	d_radix_sort_old(&d_vec, size);
	timer.Stop();
	cout << "Elapsed (old): " << timer.Elapsed() << " msecs\n";

	cudaMemcpy(h_out, d_vec, sizeof(int) * size, cudaMemcpyDeviceToHost);
	sort(h_vec, h_vec + size);
	for(int i = 0; i < size; i++){
		if(h_vec[i] != h_out[i]){
			cout << "FAILED AT " << i << "!";
			break;
		}
		if(i == size - 1)
			cout << "YESSSS!\n";
	}

	free(h_vec);
	free(h_out);
	cudaFree(d_vec);

	return 0;
}
