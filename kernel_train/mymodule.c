#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>

static int mydata __initdata = 3;

static int __init myinit(void){
	printk(KERN_INFO "Hello, world %d\n", mydata);
	return 0;
}

static void __exit myexit(void){
	printk(KERN_INFO "Goodbye, world\n");
}

module_init(myinit);
module_exit(myexit);
