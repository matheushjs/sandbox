#include <thread>
#include <iostream>
#include <queue>
#include <semaphore.h>
#include <unistd.h>

using namespace std;

queue<int> q;
sem_t items;
sem_t voids;

void producer(){
	static int count = 0;

	while(true){
		sem_wait(&voids); // Waits existence of at least 1 void
		q.push(count++);
		sem_post(&items);
	}
}

void consumer(){
	while(true){
		sem_wait(&items); // Waits existence of at least 1 item
		cout << "Queue size: " << q.size() << ". Popped: " << q.front() << "\n";
		q.pop();
		sem_post(&voids);
	}
}

int main(int argc, char *argv[]){
	sem_init(&items, 0, 0);
	sem_init(&voids, 0, 100);

	thread thr1(producer);
	thread thr2(consumer);

	thr1.join();
	thr2.join();

	return 0;
}
