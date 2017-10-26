#include <QThreadPool>
#include <QRunnable>
#include <iostream>
#include <iomanip>

using namespace std;

/* Pseudo functions */
void get_page(){
	// Calculate pi
	double sum = 0;
	for(int i = 1; i < 1E7; i++){  // i = 2
		int aux = (i+1)%2;  // 3 % 2 = 1
		aux *= 2;  // 2
		aux -= 1;  // 1
		aux = 0 - aux;  // -1
		sum += aux / (double) (2 * i - 1);
	}
	cout << setprecision(20) << 4 * sum << "\n";
}

void request_page(){
	// Should launch a new thread for getting page
	get_page();
}

int main(int argc, char *argv[]) {
	request_page();
	return 0;
}
