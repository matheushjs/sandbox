#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

#define RPC raw_pointer_cast

using std::cout;

/* Applies an AND mask to the given number 'num' shifted right by 'shift'.
 *
 * num     = 000101
 * mask    = 000011
 * shift   = 2
 * shifted = 0001
 * return  =   0001
 *         & 000011 = 1
 */
__device__
int masked_key(const unsigned int num, const unsigned int mask, const unsigned int shift){
	return (num >> shift) & mask;
}

/*
 * 'vec' is a vector of unsigned integers, with 32 bits each
 * Each element of the vector can be seen as 8 groups of 4 bits
 * We want to get a histogram of the vector, when considering each of these 8 groups separately
 * If we are focusing on the second least significant group, for example, 0xFFF0 would add up in the histogram index 15
 *
 * Since we work with values from 0 to 2^4-1, we need the 'output' vector for be 16 elements long
 * The caller is not required to set the 'output' vector to 0.
 *
 * The caller must specify which group of 4 bits they want to consider, by passing a 'shift'
 * Shift represents how much each element in the vector will be shifted left before the 4 least significant bits are read.
 * In practice, if 'shift' is 0, we consider the least significant group. If it's 4, we consider the second least significant group.
 *
 *
 * Thread Organization
 * ===
 *
 * Regarding the GPU block organization, we will accept 1D block and 1D thread organization.
 * In this organization, each thread will be responsible for taking the histogram of a section of the vector.
 * Each thread will operate directly on the 'output' vector.
 */
__global__
void d_histogram(const unsigned int * const vec, unsigned int * const output, const int shift, const int vecSize){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;

	const int totalThreads = gridDim.x * blockDim.x;     // total number of threads
	const int vecBlockSize = (vecSize / totalThreads) + 1; // +1 as failsafe

	const int initIdx = myIdx * vecBlockSize;
	const int endIdx = initIdx + vecBlockSize; // Exclusive end index

	// Reset the output vector
	int i;
	for(i = myIdx; i < 16; i += totalThreads)
		output[i] = 0;
	__syncthreads();

	for(i = initIdx; i < endIdx && i < vecSize; i++){
		const int histIdx = masked_key(vec[i], 0xF, shift);
		atomicAdd(output + histIdx, 1);
	}
}

void histogram(const unsigned int *d_vec, unsigned int *d_output, int shift, int vecSize){
	const int nThreads = 1024;
	const int nBlocks = vecSize / nThreads + 1;

	d_histogram<<<nBlocks, nThreads>>>(d_vec, d_output, shift, vecSize);
}



// Step should be 1, 2, 3... up to log2(vecSize)
__global__
void blelloch_reduce(unsigned int *d_vec, int vecSize, int step){
	const int leftDist = 1 << (step - 1);
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int myElement = (1 << step) * (myIdx + 1) - 1;

	if(myElement >= vecSize) return;
	d_vec[myElement] = d_vec[myElement] + d_vec[myElement - leftDist];
}

__global__
void blelloch_downsweep(unsigned int *d_vec, int vecSize, int step){
	const int leftDist = 1 << (step - 1);
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int myElement = (1 << step) * (myIdx + 1) - 1;

	if(myElement >= vecSize) return;

	int aux = d_vec[myElement];
	d_vec[myElement] = d_vec[myElement] + d_vec[myElement - leftDist];
	d_vec[myElement - leftDist] = aux;
}

// d_vec and vecSize must be powers of 2
void xscan(unsigned int *d_vec, int vecSize){
	int size = 1;
	int nSteps = 0;
	while(size < vecSize){
		size <<= 1; // Lowest power of 2 greater than vecSize
		nSteps++;   // The power itself
	}

	// First reduce
	for(int step = 1; step <= nSteps; step++){
		const int operationBlock = 1 << step;
		const int thrCount = vecSize / operationBlock;

		const int thrPerBlock = 1024; //Maximize threads in a block
		const int nBlocks = thrCount / thrPerBlock + 1;

		blelloch_reduce<<<nBlocks, thrPerBlock>>>(d_vec, size, step);
	}

	// Put identity value into last element
	cudaMemset(d_vec + vecSize - 1, 0, sizeof(int));

	// Now downsweep
	for(int step = nSteps; step >= 1; step--){
		const int operationBlock = 1 << step;
		const int thrCount = vecSize / operationBlock;

		const int thrPerBlock = 1024; //Maximize threads in a block
		const int nBlocks = thrCount / thrPerBlock + 1;

		blelloch_downsweep<<<nBlocks, thrPerBlock>>>(d_vec, size, step);
	}
}




// In-Place predicate taker
// 1D block and thread organization
// There is no need for more than vecSize threads allocated.
__global__
void d_make_predicate(unsigned int * const d_vec, int vecSize, int shift, int mask){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadCount = gridDim.x * blockDim.x;

	int i;
	for(i = myIdx; i < vecSize; i += threadCount)
		d_vec[i] = ((d_vec[i] >> shift) & 0xF) == mask;
}

/* This function takes in a device vector 'd_vec'
 * And places in the device vector 'd_predicates' the predicates of each element of 'd_vec'
 *
 * The predicate for an element E is [ ((E >> shift) & 0xF) == mask ]
 */
void make_predicate(const unsigned int *d_vec, int vecSize, int shift, int mask, unsigned int *d_predicates){
	int nThreads = 1024; // Maximize number of threads
	int nBlocks = vecSize / 1024 + 1;

	cudaMemcpy(d_predicates, d_vec, vecSize * sizeof(int), cudaMemcpyDeviceToDevice);

	d_make_predicate<<<nThreads, nBlocks>>>(d_predicates, vecSize, shift, mask);
}



__global__
void d_scatter_values(const unsigned int *d_input, const unsigned int *d_pred, const unsigned int *d_predScan, const unsigned int *d_histScan, unsigned int *d_output, int mask, int inputSize){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadCount = gridDim.x * blockDim.x;

	int i;
	for(i = myIdx; i < inputSize; i += threadCount){
		if(d_pred[i] == 1)
			d_output[ d_histScan[mask] + d_predScan[i] ] = d_input[i];
	}
}

void scatter_values(const unsigned int *d_input, const unsigned int *d_pred, const unsigned int *d_predScan, const unsigned int *d_histScan, unsigned int *d_output, int mask, int inputSize){
	const int nThreads = 1024;
	const int nBlocks = inputSize / nThreads + 1;

	d_scatter_values<<<nBlocks, nThreads>>>(d_input, d_pred, d_predScan, d_histScan, d_output, mask, inputSize);
}


void print_device(const unsigned int *d_vec){
	unsigned int *vec;
	vec = (unsigned int *) malloc(sizeof(int) * 32);
	cudaMemcpy(vec, d_vec, sizeof(int) * 32, cudaMemcpyDeviceToHost);
	for(int i =0; i < 32; i++)
		cout << vec[i] << " ";
	cout << "\n";
	free(vec);
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               size_t numElems)
{
	using thrust::device_vector;
	using thrust::host_vector;
	using thrust::raw_pointer_cast;

	unsigned int numElems_pow2 = 1;
	while(numElems_pow2 < numElems)
		numElems_pow2 <<= 1;

	const int nShifts = (sizeof(int) * 8); // Number of bits in an integer
	int swapCount = 0; // Number of swaps between input vec and output vec
	device_vector<unsigned int> d_histogram(32);
	device_vector<unsigned int> d_predicates(numElems_pow2);
	device_vector<unsigned int> d_predScan;

	for(int shift = 0; shift < nShifts; shift += 4){
		// Get histogram
		histogram(d_inputVals, raw_pointer_cast(d_histogram.data()), shift, numElems);

		// Scan histogram in-place
		xscan(raw_pointer_cast(d_histogram.data()), 32);

		for(int mask = 0; mask < 32; mask++){
			// Get predicates
			make_predicate(d_inputVals, numElems, shift, mask, RPC(d_predicates.data()));

			// Scan predicates in another vector
			d_predScan = d_predicates;
			xscan(raw_pointer_cast(d_predScan.data()), numElems_pow2);

			// Scatter values into output
			scatter_values( d_inputVals,
			                RPC(d_predicates.data()),
			                RPC(d_predScan.data()),
			                RPC(d_histogram.data()),
			                d_outputVals, mask, numElems);

			scatter_values( d_inputPos,
			                RPC(d_predicates.data()),
			                RPC(d_predScan.data()),
			                RPC(d_histogram.data()),
			                d_outputPos, mask, numElems);
		}

		// Switch input vector with outputvector
		std::swap(d_inputVals, d_outputVals);
		std::swap(d_inputPos, d_outputPos);
		swapCount++;
	}

	std::swap(d_inputVals, d_outputVals);
	std::swap(d_inputPos, d_outputPos);
}
