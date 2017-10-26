#include <QThreadPool>
#include <QRunnable>
#include <iostream>
#include <iomanip>

using namespace std;

class PageGetter: public QRunnable {
	int idx;
public:
	PageGetter(int index) : idx(index) {
		setAutoDelete(true);
	}
	~PageGetter(){}

	void run() {
		// Calculate pi
		double sum = 0;
		for(int i = 1; i < 1E8; i++){  // i = 2
			int aux = (i+1)%2;  // 3 % 2 = 1
			aux *= 2;  // 2
			aux -= 1;  // 1
			aux = 0 - aux;  // -1
			sum += aux / (double) (2 * i - 1);
		}
		cout << idx << ": " << setprecision(20) << 4 * sum << "\n";
	}
};


QThreadPool pool;
int cnt = 0;

void request_page(){
	pool.start(new PageGetter(cnt));
	cout << "Started: " << cnt++ << "\n";
}

int main(int argc, char *argv[]) {
	pool.setMaxThreadCount(20);
	cout << "Max: " << pool.maxThreadCount() << "\n";

	while(true){
		char n;
		cin >> n; // waits any input
		request_page();
	}
	return 0;
}
