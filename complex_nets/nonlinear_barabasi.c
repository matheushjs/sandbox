#include <stdlib.h>
#include <stdio.h>
#include <igraph/igraph.h>

int main(int argc, char *argv[]){
	igraph_t g;
	igraph_vector_t v;
	int i;

	if(argc != 4){
		printf("Usage: %s N m power\n", argv[0]);
		return 1;
	}

	int N = atoi(argv[1]);
	int m = atoi(argv[2]);
	double power = atof(argv[3]);

	igraph_barabasi_game(&g, N, /*power=*/ power, m, 0, 0, /*A=*/ 1, 1, 
		IGRAPH_BARABASI_PSUMTREE, /*start_from=*/ 0);

	igraph_vector_init(&v, 0);
	igraph_get_edgelist(&g, &v, 0);
	for(i=0; i<igraph_ecount(&g); i++){
		printf("%d %d\n", (int) VECTOR(v)[2*i], (int) VECTOR(v)[2*i + 1]);
	}

	igraph_vector_destroy(&v);
	igraph_destroy(&g);

	return 0;
}
