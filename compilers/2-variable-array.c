#include <stdio.h>

void main(int argc, char *argv[]){
	int array[argc];
	array[argc-1] = 72;
	printf("%d\n", array[argc-1]);
}
