#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
	// We will assign each row to a block
	// And each pixel in that row to a thread
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int idx = row * numCols + col;

    greyImage[idx] = 0.299f * rgbaImage[idx].x;
    greyImage[idx] += 0.587f * rgbaImage[idx].y;
    greyImage[idx] += 0.114f * rgbaImage[idx].z;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 blockSize(numCols, 1, 1);
  const dim3 gridSize(numRows, 1, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
