#include <stdio.h>

/* Utilizamos este código para analisar diferenças arquiteturais relativas
 *   a códigos com dependência de dados.
 *
 * O algoritmo utilizado faz parte de uma classe de algoritmos comumente chamada
 *   de "Stencil".
 * Algoritmos de "Stencil" consistem em, dado uma matriz N-dimensional, cada elemento
 *   dessa matriz recebe o valor de uma computação (e.g. soma) sobre elementos da mesma
 *   matriz, sendo que esses elementos são acessados de acordo com um padrão.
 *
 * Em geral, algoritmos de Stencil recebem uma matriz de entrada e criam uma nova matriz
 *   que será retornada ao usuário. No entanto, para analizar dependências de dados, aqui
 *   realizamos a operação de stencil na própria matriz (no caso unidimensional, i.e. um
 *   vetor), sem se preocupar com a corretude ou a funcionalidade do algoritmo.
 *
 * Neste código, a operação de stencil realizada é a seguinte:
 *   vector[i] = (vector[i-2] + vector[i-1] + vector[i] + vector[i+1] + vector[i+2]) / 5
 * Ou seja, cada elemento recebe a média dos 5 elementos que o circundam.
 */

#define N 10000

int main(int argc, char *argv[]){
	int i, j;
	int *vector = malloc(sizeof(int) * N);

	// Inicializa vetor
	for(i = 0; i < N; i++){
		vector[i] = i;
	}

	// Realiza computação
	for(j = 2; j < N-2; j++){
		vector[j] = (vector[j-2] + vector[j-1] + vector[j] + vector[j+1] + vector[j+2]) / 5;
	}

	// Imprime alguns elementos
	for(i = 0; i < 8; i++){
		printf("%d ", vector[8*i]);
	}
	printf("\n");

	return 0;
}
