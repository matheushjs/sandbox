#include <stdio.h>
#include <functional>

using namespace std;

int main(){
	int a = 10;

	auto lambda = [=](int b){
		printf("%d %d\n", a, b);
	};
	
	lambda(5); // Imprime '10 5'
}
