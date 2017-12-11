#include <cuda.h>
#include "blelloch_scan.h"

// pow2(X) returns 2^X
#define pow2(X) (1 << X)

// Step should be 1, 2, 3... up to log2(vecSize) (round up)
// That's because the number of steps required is known once you have the size of the vector
// Also, the current step value determines what each thread will do at each point
// Number of threads required per step:
//    step1:   vecSize/2
//    step2:   vecSize/4
// and so on.
__global__
void blelloch_reduce(unsigned int *d_vec, int vecSize, int step){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int myElem = pow2(step) * (myIdx + 1) - 1;
	const int otherElem = myElem - pow2(step - 1);

	if(myElem >= vecSize) return;
	d_vec[myElem] = d_vec[myElem] OPERATOR d_vec[otherElem];
}

// Be nSteps = log2(vecSize) (round up)
// Step should be nSteps, nSteps-1, nSteps-2, ..., 1
// That's because the downsweep element accessing pattern is the opposite of the reduce pattern
// Number of threads required per step:
//    step nStep:   vecSize/pow(nStep)
//    step nStep-1: vecSize/pow(nStep-1)
//    ...
//    step2:   vecSize/4
//    step1:   vecSize/2
__global__
void blelloch_downsweep(unsigned int *d_vec, int vecSize, int step){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int myElem = pow2(step) * (myIdx + 1) - 1;
	const int otherElem = myElem - pow2(step - 1);

	if(myElem >= vecSize) return;

	int aux = d_vec[myElem];
	d_vec[myElem] = d_vec[myElem] OPERATOR d_vec[otherElem];
	d_vec[otherElem] = aux;
}

// Size of d_vec must be a power of 2.
void xscan(unsigned int *d_vec, int vecSize, cudaStream_t st){
	int size = 1;
	int nSteps = 0;
	while(size < vecSize){
		size <<= 1; // Lowest power of 2 greater than or equal to vecSize
		nSteps++;   // The power itself
	}

	// First reduce
	for(int step = 1; step <= nSteps; step++){
		const int thrCount = vecSize / pow2(step);
		const int thrPerBlock = 256; //Reasonable number of threads in a block
		const int nBlocks = thrCount / thrPerBlock + 1;

		blelloch_reduce<<<nBlocks, thrPerBlock, 0, st>>>(d_vec, size, step);
	}

	// Put identity value into last element
	unsigned int aux = IDENTITY;
	cudaMemcpy( &d_vec[vecSize - 1], &aux, sizeof(int), cudaMemcpyHostToDevice);

	// Now downsweep
	for(int step = nSteps; step >= 1; step--){
		const int thrCount = vecSize / pow2(step);
		const int thrPerBlock = 256; //Reasonable number of threads in a block
		const int nBlocks = thrCount / thrPerBlock + 1;

		blelloch_downsweep<<<nBlocks, thrPerBlock, 0, st>>>(d_vec, size, step);
	}
}

/*
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#define RPC raw_pointer_cast

using thrust::device_vector;
using thrust::host_vector;

bool check(host_vector<unsigned int> &in, host_vector<unsigned int> &out){
	int sum = IDENTITY;
	bool ok = true;
	for(int i = 0; i < in.size(); i++){
		if(sum != out[i]) ok = false;
		sum = sum OPERATOR in[i];
	}
	return ok;
}

int main(int argc, char *argv[]){
	using std::cout;

	host_vector<unsigned int> h_vec(pow2(20));
	for(int i = 0; i < h_vec.size(); i++)
		h_vec[i] = h_vec.size() - i;

	device_vector<unsigned int> d_vec = h_vec;
	device_vector<unsigned int> d_out(h_vec.size());

	xscan(RPC(d_vec.data()), d_vec.size());

	host_vector<unsigned int> h_out = d_vec;

	for(int i = 0; i < 32; i++)
		cout << h_out[i] << " ";
	cout << "\n";

	cout << (check(h_vec, h_out) ? "Scanned\n" : "Messed up!\n");

	return 0;
}
*/
