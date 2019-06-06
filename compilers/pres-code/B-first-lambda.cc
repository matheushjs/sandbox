#include <functional>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]){
	int a = 5;

	auto lambda = [&](int b){
		cout << "a, b == " << a << ", " << b << "\n";
	};

	lambda(10);

	return 0;
}
