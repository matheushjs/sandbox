#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NITER 1000000

int cnt = 0;
pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

void *count(void *a){
	int i;

	for(i = 0; i < NITER; i++){
		pthread_mutex_lock(&mut);
		cnt++;
		pthread_mutex_unlock(&mut);
	}
}

int main(int argc, char * argv[]){
	pthread_t tid;

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

	pthread_mutex_destroy(&mut);

	pthread_exit(NULL);
}
