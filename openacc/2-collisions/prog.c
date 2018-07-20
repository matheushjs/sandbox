#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int x;
	int y;
	int z;
} int3;

int3 *create_vector(int size){
	int i;
	int3 *result = (int3 *) malloc(sizeof(int3) * size);

	for(i = 0; i < size; i++){
		result[i].x = 0;
		result[i].y = 0;
		result[i].z = (i % (size/2)) * 4 - size;
	}

	printf("Collisions expected: %d\n", size/2);

	return result;
}

int count(int3 *vec, int vecSize){
	int i, j;
	int colls;

	#pragma acc data copyin(vec[vecSize]) copyout(colls)
	{
		colls = 0;
		
		#pragma acc parallel loop private(j) reduction( +:colls )
		for(i = 0; i < (vecSize-1); i++){
			for(j = i+1; j < vecSize; j++){
				if(
					vec[i].x == vec[j].x
					&& vec[i].y == vec[j].y
					&& vec[i].z == vec[j].z
				){
					colls++;
				}
			}
		}
	}

	return colls;
}

void t2(int vecSize){
	int3 *vec = create_vector(vecSize);

	int colls = count(vec, vecSize);

	printf("%d\n", colls);

	free(vec);
}

int main(int argc, char *argv[]){
	int vecSize;
	
	switch(argc){
		case 1:
			vecSize =  128 * 1024;
			break;
		case 2:
			vecSize = atoi(argv[1]);
			break;
		case 3:
			fprintf(stderr, "Usage: %s [problem_size]\n", argv[0]);
			return 1;
	}
	
	t2(vecSize);
	
	return 0;
}
