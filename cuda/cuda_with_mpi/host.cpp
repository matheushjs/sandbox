#include <mpi.h>
#include "device.h"

int main(int argc, char *argv[]){
	MPI_Init(NULL, NULL);

	do_something(10);

	MPI_Finalize();
	return 0;
}
