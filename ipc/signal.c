#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <sys/types.h>

void sighandler(int a){
	printf("Hello World!\n");
}

int main(int argc, char *argv[]){
	int pid = fork();

	if(pid == 0){
		sleep(1);
		kill(pid, SIGALRM);
		sleep(2);
	} else {
		signal(SIGALRM, sighandler);

		int i;
		for(i = 0; i < (int) 1E9; i++){
			if(i % (int) 1E8 == 0){
				printf("%d\n", i);
			}
		}
	}

	return 0;
}
