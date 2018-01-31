obj-m += mymodule.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

test:
	sudo insmod mymodule.ko
	sudo rmmod mymodule.ko
	tail -15 /var/log/kern.log
