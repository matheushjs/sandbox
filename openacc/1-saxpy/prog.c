#include <stdio.h>
#include <stdlib.h>

// Y = a*X + Y
void saxpy_parallel(int n, float a, float *x, float *y) {
	int i;

	#pragma acc kernels
	for (i = 0; i < n; i++){
		y[i] = a*x[i] + y[i];
	}
}

int main(int argc, char *argv[]){
	float a = 5.0f;
	float X[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
	float Y[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	int n = sizeof(X) / sizeof(float);
	int i;

	saxpy_parallel(n, a, X, Y);
	
	for(i = 0; i < n; i++){
		printf("%f ", Y[i]);
	}
	printf("\n");

	return 0;
}
