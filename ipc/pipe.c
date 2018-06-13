#include <unistd.h>
#include <stdio.h>

int main(int argc, char *argv[]){
	int pipefd[2];

	pipe(pipefd);

	int pid = fork();

	if(pid == 0){
		char buf[50] = "Hello World\n";
		FILE *fp = fdopen(pipefd[1], "w");
		fwrite(buf, sizeof(char), 50, fp);
	} else {
		char buf[50];
		FILE *fp = fdopen(pipefd[0], "r");
		fread(buf, sizeof(char), 50, fp);
		printf("%s\n", buf);
	}

	return 0;
}
