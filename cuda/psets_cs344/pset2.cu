#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
	const uint2 myCoord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
	                                 blockIdx.y * blockDim.y + threadIdx.y);
	const int myIdx = myCoord.y * numCols + myCoord.x;

	if(myCoord.x >= numCols || myCoord.y >= numRows)
		return;

	// get coordinates for top-left pixel
	const int offset = filterWidth / 2;
	const uint2 topLeft = make_uint2(myCoord.x - offset, myCoord.y - offset);
	int curIdx = topLeft.y * numCols + topLeft.x;

	unsigned int i, j;
	float sum = 0, weights = 0;
	for(i = 0; i < filterWidth; i++){
		for(j = 0; j < filterWidth; j++){
			if( (topLeft.x + j) < numCols && (topLeft.y + i) < numRows){
				weights += filter[i*filterWidth + j];
				sum += filter[i*filterWidth + j] * inputChannel[curIdx];
			}

			curIdx++;
		}
		curIdx -= filterWidth; // Rewind to left
		curIdx += numCols; // Jump a row
	}
	
	outputChannel[myIdx] = (unsigned int) (sum / weights);
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
	const int myRow = blockIdx.x;
	const int myCol = threadIdx.x;
	const int idx = myRow * numCols + myCol;

	redChannel[idx] = inputImageRGBA[idx].x;
	greenChannel[idx] = inputImageRGBA[idx].y;
	blueChannel[idx] = inputImageRGBA[idx].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}


// Some globals
unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;


void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));

  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(16, 16);
  const dim3 gridSize(numCols/16 + 1, numRows/16 + 1);

  separateChannels<<<numRows, numCols>>> (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
