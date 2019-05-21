#include <functional>
#include <stdio.h>

using namespace std;

void normal_func(function<void(void)> f){
	f();
}

int main(int argc, char *argv[]){
	int n = 5;
	function<void(void)> lambda = [=](){
		printf("%d\n", n);
	};
	normal_func(lambda);
}
