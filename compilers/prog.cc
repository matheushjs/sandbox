#include <iostream>
#include <functional>

using namespace std;

void print(function<int(void)> f){
	cout << f() << "\n";
}

int main(int argc, char *argv[]){
	auto lambda = [argc](){
		return argc;
	};

	print(lambda);

	return 0;
}
