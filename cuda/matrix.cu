#include <stdio.h>

#define DIM 32 // 32 is maximum for now

int *create_matrix(int row, int col){
	return (int *) malloc(sizeof(int) * row * col);
}

void print_matrix(int *mat, int row, int col){
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			printf("%d ", mat[i*col + j]);
		}
		printf("\n");
	}
}

void fill_matrix(int *mat, int row, int col, int fill){
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			mat[i*col + j] = fill;
		}
	}
}

// Kernel that uses a 2D thread structure for squaring the matrix
// Only one block is allowed (shared memory)
__global__
void matrix_square(int *mat, int dim){
	const int myRow = threadIdx.y;
	const int myCol = threadIdx.x;
	const int myPos = myRow * dim + myCol;
	const int colStride = dim;

	int i, j;
	const int rowBase = myRow * dim; // Left matrix starting position (A x A)
	const int colBase = myCol;       // Right matrix starting position
	int sum = 0;
	for(i = 0; i < dim; i++){
		for(j = 0; j < dim; j++){
			sum += mat[rowBase + i] * mat[colBase + colStride * j];
		}
	}
	__syncthreads();
	mat[myPos] = sum;
}

int main(int argc, char *argv[]){
	int *h_mat = create_matrix(DIM, DIM);
	fill_matrix(h_mat, DIM, DIM, 2);

	const size_t size = DIM*DIM*sizeof(int);
	int *d_mat;
	cudaMalloc(&d_mat, size);
	cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);
	
	matrix_square<<<1,dim3(DIM,DIM)>>>(d_mat, DIM);
	
	cudaMemcpy(h_mat, d_mat, size, cudaMemcpyDeviceToHost);
	print_matrix(h_mat, DIM, DIM);
	cudaFree(d_mat);
	free(h_mat);
	return 0;
}
