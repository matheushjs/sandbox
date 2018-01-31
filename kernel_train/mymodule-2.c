#include <linux/module.h>
#include <linux/kernel.h>

static void __exit my_exit(void){
	printk(KERN_INFO "Goodbye, world\n");
}

module_exit(my_exit);
