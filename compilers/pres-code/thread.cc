#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

void f1(){
	int a = 10;

	thread([&](){
		sleep(1);
		cout << a << "\n";
	}).detach();
}

int main(int argc, char *argv[]){
	f1();

	sleep(5);
	return 0;
}
