#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
	int i, j;
	int *vector = malloc(sizeof(int) * 8);

	// Inicializa vetor
	for(i = 0; i < 8; i++){
		vector[i] = 1;
	}

	// Realiza computação
	for(i = 0; i < 50000000; i++){
		for(j = 0; j < 4; j++){
			vector[j] = vector[j] + vector[j+1] + vector[j+2] + vector[j+3];
		}
	}

	// Imprime vetor
	for(i = 0; i < 4; i++){
		printf("%d ", vector[i]);
	}
	printf("\n");

	return 0;
}
