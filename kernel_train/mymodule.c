
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/uaccess.h>


/* Declarations */
int init_module(void);
void cleanup_module(void);
static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_read(struct file *, char *, size_t, loff_t *);
static ssize_t device_write(struct file *, const char *, size_t, loff_t *);

#define SUCCESS 0
#define DEVICE_NAME "mymodule"
#define BUF_LEN 80

static int g_major;
static int g_deviceOpen = 0;

static char g_msg[BUF_LEN];
static char *g_msgPtr;

static struct file_operations fops = {
	.read = device_read,
	.write = device_write,
	.open = device_open,
	.release = device_release
};

int init_module(void){
	g_major = register_chrdev(0, DEVICE_NAME, &fops);

	if(g_major < 0){
		printk(KERN_ALERT "Failed registering device. Received major: %d\n", g_major);
		return g_major;
	}

	printk(KERN_INFO "Please run command: 'mknod /dev/%s c %d 0'.\n", DEVICE_NAME, g_major);
	printk(KERN_INFO "Later, remove device node with command: 'rm /dev/%s'.\n", DEVICE_NAME);

	return SUCCESS;
}

void cleanup_module(void){
	unregister_chrdev(g_major, DEVICE_NAME);
}

static int device_open(struct inode *inode, struct file *file){
	static int counter = 0;

	if(g_deviceOpen)
		return -EBUSY;

	g_deviceOpen = 1;
	sprintf(g_msg, "Counter: %d\n", counter++);
	g_msgPtr = g_msg;
	try_module_get(THIS_MODULE);

	return SUCCESS;
}

static int device_release(struct inode *inode, struct file *file){
	g_deviceOpen = 0;
	module_put(THIS_MODULE);
	return 0;
}

static ssize_t device_read(struct file *filp, char *buffer, size_t length, loff_t *offset){
	int bytesRead = 0;
	
	if(*g_msgPtr == '\0')
		return 0;

	while(length > 0 && *g_msgPtr != '\0'){
		put_user(*g_msgPtr, buffer);
		g_msgPtr++;
		buffer++;
		length--;
		bytesRead++;
	}

	return bytesRead;
}

static ssize_t device_write(struct file *filp, const char *buffer, size_t len, loff_t *off){
	printk(KERN_ALERT "Write not supported.\n");
	return -EINVAL;
}
