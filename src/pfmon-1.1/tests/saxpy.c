#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>

#define VECTOR_SIZE	1000000

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}


static void
saxpy(unsigned long *a, unsigned long *b, unsigned long *c, unsigned long size)
{
	unsigned long i;

	for(i=0; i < size; i++) {
		c[i] = 2*a[i] + b[i];
	}
}

static void
saxpy2(unsigned long *a, unsigned long *b, unsigned long *c, unsigned long size)
{
	unsigned long i;

	for(i=0; i < size; i++) {
		c[i] = 2*a[i] + b[i];
	}
}


int
main(int argc, char **argv)
{
	unsigned long size;
	unsigned long *a, *b, *c;

	size = argc > 1 ? strtoul(argv[1], NULL, 0) : VECTOR_SIZE;

	printf("%lu entries = %lu bytes/vector = %lu Mbytes total\n", 
		size, 
		size*sizeof(unsigned long),
		(3*size*sizeof(unsigned long))>>20
	);

	a = malloc(size*sizeof(unsigned long));
	b = malloc(size*sizeof(unsigned long));
	c = malloc(size*sizeof(unsigned long));

	if (a == NULL || b == NULL || c == NULL)
		fatal_error("Cannot allocate vectors\n");

	saxpy(a, b, c, size);

	saxpy2(a, b, c, size);

	return 0;
}

