#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>

static int __init my_init(void){
	printk(KERN_INFO "Hello, world\n");
	return 0;
}

static void __exit my_exit(void){
	printk(KERN_INFO "Goodbye, world\n");
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Matheus H. J. Saldanha <m@gmail.com>");
MODULE_DESCRIPTION("A simple device");
MODULE_SUPPORTED_DEVICE("something");
