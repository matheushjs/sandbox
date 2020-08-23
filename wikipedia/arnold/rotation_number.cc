#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define WIDTH (8*500)
#define HEIGHT (8*500)
#define ITER_COUNT 4000
#define ITER_TO_IGNORE 3800

static float pixels[HEIGHT][WIDTH] = { 0.0f };

using namespace std;

static const long double PI = atanl(1) * (long double) 4;

inline
long double F(long double theta, long double omega, long double K){
	return theta + omega + (K / 2 / PI) * sinl(2 * PI * theta);
}

void generate_file(){
	#pragma omp parallel for
	for(size_t i = 0; i < HEIGHT; i++){
		for(size_t j = 0; j < WIDTH; j++){
			long double theta = rand() / (long double) RAND_MAX;
			double aux;

			long double K = (i / (long double) HEIGHT) * 2 * PI;
			long double omega = (j / (long double) WIDTH);

			long double avgRot = 0;
			for(size_t k = 0; k < ITER_COUNT; k++){
				theta = F(theta, omega, K);

				if(k >= ITER_TO_IGNORE){
					long double rot = theta / k;
					avgRot += rot / (ITER_COUNT - ITER_TO_IGNORE);
				}
			}

//			printf("%lf ", modf(theta, &aux));
			//double widthPixel = modf(theta + 50.5, &aux) * WIDTH;
			float &pixel = pixels[i][j];
			// pixel += ((1<<8 - 1) - pixel) / 1.3;
			pixel = (float) fabsl(avgRot);
		}
		printf("Row: %lu / %d\n", i+1, HEIGHT);
	}

	FILE *fp = fopen("pixels.mat", "w+");
	fwrite(pixels, sizeof(pixels), 1, fp);
	fclose(fp);
}

int main(int argc, char *argv[]){
	generate_file();
	return 0;
}
