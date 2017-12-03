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
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	unrelated_stuff();
}
