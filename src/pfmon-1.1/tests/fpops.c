#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>

int
main(int argc, char **argv)
{
	unsigned long count;
	double f = 0.0, g = 1.0;

	count = argc > 1 ? strtoul(argv[1], NULL, 0) : 1;


	while (count--) {
		f = f + 2.0*g;
	}
	printf("f=%g\n", f);

	return 0;
}

