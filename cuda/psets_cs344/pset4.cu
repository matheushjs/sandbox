#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using std::cout;

__device__
void swap(unsigned int *a, unsigned int *b){
	unsigned int aux;
	if(*a > *b){
		aux = *a;
		*a = *b;
		*b = aux;
	}
}

__global__
void oddeven_sort(unsigned int * const d_vec, int vecSize){
	int init = threadIdx.x * 2; // First element to be analyzed by this thread
	int stride = blockDim.x * 2;

	int step;
	int increment = 1;
	for(step = 0; step < vecSize; step++){
		int curElem;
		for(curElem = init; curElem < vecSize - 1; curElem += stride){
			swap(d_vec + init, d_vec + init + 1);
		}

		init += increment;
		increment *= -1; // Next iteration, undo the sum on init
	}
}

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

/* SUMMARY
 * ===
 *
 * 'vec' is a vector of unsigned integers, with 32 bits each
 * Each element of the vector can be seen as 4 groups of 8 bits
 * We want to get a histogram of the vector, when considering each of these 4 groups separately
 * If we are focusing on the second least significant group, for example, 0xFF00 would add up in the histogram index 255
 *
 * Since we work with values from 0 to 2^8-1, we need the 'output' vector for be 256 elements long
 * The caller is not required to set the 'output' vector to 0.
 *
 * The caller must specify which group of 8 bits they want to consider, by passing a 'shift'
 * Shift represents how much each element in the vector will be shifted left before the 8 least significant bits are read.
 * In practice, if 'shift' is 0, we consider the least significant group. If it's 8, we consider the second least significant group.
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
	for(i = myIdx; i < 256; i += totalThreads)
		output[i] = 0;
	__syncthreads();

	for(i = initIdx; i < endIdx && i < vecSize; i++){
		const int histIdx = masked_key(vec[i], 0xFF, shift);
		atomicAdd(output + histIdx, 1);
	}
}

void unrelated_stuff(){
	thrust::host_vector<unsigned int> h_vec(16);

	for(int i = 0; i < 16; i++)
		h_vec[i] = 16 - i;

	thrust::device_vector<unsigned int> d_vec = h_vec;

	oddeven_sort<<<1, 8>>>(thrust::raw_pointer_cast(d_vec.data()), 16);

	h_vec = d_vec;

	for(int i = 0; i < 16; i++)
		cout << h_vec[i] << " ";
	cout << "\n";

	for(int i = 0; i < 16; i++)
		h_vec[i] = 5;
	d_vec = h_vec;
	thrust::device_vector<unsigned int> d_hist(256);
	histogram<<<1, 16>>>(thrust::raw_pointer_cast(d_vec.data()),
	                thrust::raw_pointer_cast(d_hist.data()),
	                0, 16);

	thrust::host_vector<unsigned int> h_hist = d_hist;
	for(int i = 0; i < 256; i++)
		cout << h_hist[i] << " ";
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
