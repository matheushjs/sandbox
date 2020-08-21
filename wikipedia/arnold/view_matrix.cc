#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

#define SCALE (1/1) // WIDTH / HEIGHT

using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
	FILE *fp = fopen("pixels.mat", "r");
	fseek(fp, 0, SEEK_END);
	unsigned int floatCount = ftell(fp) / sizeof(float);

	fseek(fp, 0, SEEK_SET);
	float *ptr = (float *) malloc(sizeof(float) * floatCount);
	fread(ptr, floatCount, sizeof(float), fp);
	fclose(fp);

	for(size_t i = 0; i < floatCount; i++){
		if(ptr[i] > 1.643345)
			ptr[i] = 1.643345;
	}

	unsigned int height = sqrt(floatCount / SCALE);
	unsigned int width  = floatCount / height;
	Mat grey(height, width, CV_32FC1, ptr);
	flip(grey, grey, 0);
	grey = -grey;

	//grey = -grey; // To invert colors
	normalize(grey, grey, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat color(height, width, CV_8UC3, Scalar::all(0));
	applyColorMap(grey, color, COLORMAP_OCEAN);

	imwrite("rotation_number.jpg", color);

	return 0;
}
