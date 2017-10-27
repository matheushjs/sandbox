#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace std;

volatile int waiting;
volatile bool interested[2] = { false, false };

void enter_critical(int myID){
	int other = 1 - myID;
	interested[myID] = true;
	waiting = myID;
	while(interested[other] == true && waiting == myID);
}

void leave_critical(int myID){
	interested[myID] = false;
}

vector<int> vec;

void producer(){
	static int count = 0;
	while(true){
		usleep(1000 * 20);
		enter_critical(0);
		usleep(1000 * 20);

		if(vec.size() < 100)  // Critical region
			vec.push_back(count++);

		leave_critical(0);
	}
}

void consumer(){
	while(true){
		usleep(1000 * 20);
		enter_critical(1);
		usleep(1000 * 20);

		if(vec.size() > 0){
			cout << vec.back() << "\n";
			vec.pop_back();
		}

		leave_critical(1);
	}
}

int main(int argc, char *argv[]){
	thread thr1(producer);
	thread thr2(consumer);

	thr1.join();
	thr2.join();

	return 0;
}
