#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

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
void histogram(const unsigned int * const vec, unsigned int * const output, const int shift, const int vecSize){
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

unsigned int *xscan(const unsigned int *vec, int vecSize){
	int size = 1;
	int nSteps = 0;
	while(size < vecSize){
		size <<= 1; // Lowest power of 2 greater than vecSize
		nSteps++;   // The power itself
	}

	unsigned int *d_vec;
	cudaMalloc(&d_vec, sizeof(int) * size);
	cudaMemcpy(d_vec, vec, sizeof(int) * vecSize, cudaMemcpyHostToDevice);

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

	unsigned int *result = (unsigned int *) malloc(sizeof(int) * vecSize);
	cudaMemcpy(result, d_vec, sizeof(int) * vecSize, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);

	return result;
}




// In-Place predicate taker
// 1D block and thread organization
// There is no need for more than vecSize threads allocated.
__global__
void d_make_predicate(unsigned int * const d_vec, int vecSize, int shift){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadCount = gridDim.x * blockDim.x;

	int i;
	for(i = myIdx; i < vecSize; i += threadCount)
		d_vec[i] = (d_vec[i] >> shift) & 0xF;
}

/* This function takes in a device vector 'd_vec'
 * And places in the device vector 'd_predicates' the predicates of each element of 'd_vec'
 *
 * The predicate for an element E is [ (E >> shift) & 0xF ]
 */
void make_predicate(const unsigned int *d_vec, int vecSize, int shift, unsigned int *d_predicates){
	int nThreads = 1024; // Maximize number of threads
	int nBlocks = vecSize / 1024 + 1;

	cudaMemcpy(d_predicates, d_vec, vecSize * sizeof(int), cudaMemcpyDeviceToDevice);

	d_make_predicate<<<nThreads, nBlocks>>>(d_predicates, vecSize, shift);
}

void unrelated_stuff(){
	using namespace thrust;

	host_vector<unsigned int> h_vec(4);
	h_vec[0] = 512;
	h_vec[1] = 64;
	h_vec[2] = 32;
	h_vec[3] = 2;

	device_vector<unsigned int> d_vec = h_vec;
	device_vector<unsigned int> d_pred(4);

	make_predicate(raw_pointer_cast(d_vec.data()), 4, 4, raw_pointer_cast(d_pred.data()));

	h_vec = d_pred;
	for(int i = 0; i < 4; i++)
		cout << h_vec[i] << " ";
	cout << "\n";
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	unrelated_stuff();
}
