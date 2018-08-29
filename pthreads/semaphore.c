#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>

#define NITER 1000000

int cnt = 0;
sem_t sem;

void *count(void *a){
	int i;

	for(i = 0; i < NITER; i++){
		sem_wait(&sem);
		cnt++;
		sem_post(&sem);
	}
}

int main(int argc, char * argv[]){
	pthread_t tid;

	sem_init(&sem, 0, 1);

	if(pthread_create(&tid, NULL, count, NULL)){
		printf("\n ERROR creating thread 2");
		exit(1);
	}

	count(NULL);

	if(pthread_join(tid, NULL)){
		printf("\n ERROR joining thread");
		exit(1);
	}

	if(cnt < 2*NITER)
		printf("\n BOOM! cnt is [%d], should be %d\n", cnt, 2*NITER);
	else printf("\n OK! cnt is [%d]\n", cnt);

	sem_destroy(&sem);

	pthread_exit(NULL);
}
