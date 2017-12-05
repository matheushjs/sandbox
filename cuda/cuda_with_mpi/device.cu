// vim: filetype=cpp
#include <cuda.h>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__
void sum(int *vec1, int *vec2, int vecSize){
	const int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
	const int thrCount = gridDim.x * blockDim.x;

	int i;
	for(i = myIdx; i < vecSize; i += thrCount)
		vec1[i] = vec1[i] + vec2[i];
}

using namespace thrust;

void do_something(int i){
	host_vector<int> vec1(32 * 1024 * 1024);
	host_vector<int> vec2(32 * 1024 * 1024);

	for(int i = 0; i < vec1.size(); i++){
		vec1[i] = i;
		vec2[i] = vec2.size() - i;
	}

	device_vector<int> d_vec1 = vec1;
	device_vector<int> d_vec2 = vec2;

	sum<<<1,1024>>>(
			raw_pointer_cast(d_vec1.data()),
			raw_pointer_cast(d_vec2.data()),
			d_vec1.size());
	
	vec1 = d_vec1;

	for(int i = 0; i < vec1.size(); i += 1024 * 1024)
		printf("%d ", vec1[i]);
	printf("\n");
}
