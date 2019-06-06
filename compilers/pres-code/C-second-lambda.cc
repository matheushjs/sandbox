#include <functional>
#include <stdio.h>

using namespace std;

void func(function<void(void)> f){
	f();
}

int main(int argc, char *argv[]){
	int n = 5;

	auto lambda = [&](){
		putchar(n + '0');
	};

	func(lambda);

	return 0;
}
