#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#define WIDTH 1000
#define HEIGHT 2000
#define ITER_COUNT 8000
#define ITER_TO_IGNORE 1000

static unsigned char pixels[HEIGHT][WIDTH] = { 0 };

using namespace cv;
using namespace std;

static const long double PI = atanl(1) * (long double) 4;

inline
long double F(long double theta, long double omega, double K){
	return theta + omega + (K / 2 / PI) * sinl(2 * PI * theta);
}

int main(int argc, char *argv[]){
	for(size_t i = 0; i < HEIGHT; i++){
		//for(size_t j = 0; j < WIDTH; j++){
			long double theta = rand() / (long double) RAND_MAX;
			long double omega = 1 / (long double) 3;
			double K = (i / (double) HEIGHT) * 4 * PI;
			double aux;

			for(size_t k = 0; k < ITER_COUNT; k++){
				theta = F(theta, omega, K);

				if(k > ITER_TO_IGNORE){
					double widthPixel = modf(theta + 50.5, &aux) * WIDTH;
					unsigned char &pixel = pixels[HEIGHT - i - 1][ (size_t) widthPixel];
					// pixel += ((1<<8 - 1) - pixel) / 1.3;
					pixel = min(pixel + (int) (15 * exp(-2 * pixel / 255)), 255);
				}
			}

//			printf("%lf ", modf(theta, &aux));
		//}
	}

	Mat grey(HEIGHT, WIDTH, CV_8UC1, pixels);
	Mat color(HEIGHT, WIDTH, CV_8UC3, Scalar::all(0));
	normalize(color, color, 0, 255, NORM_MINMAX, CV_8UC1);
	applyColorMap(grey, color, COLORMAP_INFERNO);
	//image = imread( argv[1], 1 );
	//namedWindow("Display Image", WINDOW_FREERATIO);
	//imshow("Display Image", image);
	imwrite("bifurcation_diagram.jpg", color);
	waitKey(0);

	printf("%10.60LG\n", PI);

	return 0;
}
