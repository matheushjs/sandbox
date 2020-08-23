#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define SCALE (1/1) // WIDTH / HEIGHT

using namespace cv;
using namespace std;

Mat colorbar(size_t height, float from, float to, float by){
	size_t width = 0.05 * height;

	float *ptr = (float *) malloc(height * width * sizeof(float));
	for(size_t i = 0; i < height; i++){
		for(size_t j = 0; j < width; j++){
			float *aux = ptr + (i*width + j);
			*aux = (i / (double) height) * 255.0;
		}
	}

	Mat grey(height, width, CV_32FC1, ptr);
	flip(grey, grey, 0);
	grey = -grey;
	normalize(grey, grey, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat color(height, width, CV_8UC3, Scalar::all(0));
	applyColorMap(grey, color, COLORMAP_OCEAN);

	Mat color2(height, width, CV_8UC3, Scalar::all(255));

	size_t steps = (to - from) / by + 1;
	for(size_t i = 0; i < steps; i++){
		stringstream stream;
		stream << setprecision(4) << from + by*(steps - i - 1);
		putText(color2,
				stream.str(),
				Point(0.1*width, 0.03*height + i * (0.96*height) / (steps - 1)),
				FONT_HERSHEY_SIMPLEX,
				6,
				Scalar(0, 0, 0),
				8);
	}

	Mat white(height, 0.5*width, CV_8UC3, Scalar::all(255));
	hconcat(color, color2, color);
	hconcat(white, color, color);

	return color;
}

int main(int argc, char *argv[]){
	FILE *fp = fopen("pixels.mat", "r");
	fseek(fp, 0, SEEK_END);
	unsigned int floatCount = ftell(fp) / sizeof(float);
	unsigned int height = sqrt(floatCount / SCALE);
	unsigned int width  = floatCount / height;

	Mat bar = colorbar(height, 0, 1.6, 0.1);

	fseek(fp, 0, SEEK_SET);
	float *ptr = (float *) malloc(sizeof(float) * floatCount);
	fread(ptr, floatCount, sizeof(float), fp);
	fclose(fp);

	for(size_t i = 0; i < floatCount; i++){
		if(ptr[i] > 1.643345)
			ptr[i] = 1.643345;
	}

	Mat grey(height, width, CV_32FC1, ptr);
	flip(grey, grey, 0);
	grey = -grey;

	//grey = -grey; // To invert colors
	normalize(grey, grey, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat color(height, width, CV_8UC3, Scalar::all(0));
	applyColorMap(grey, color, COLORMAP_OCEAN);

	hconcat(color, bar, color);
	imwrite("rotation_number.jpg", color);

	return 0;
}
